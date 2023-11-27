"""q0-based significance search."""
import os
import glob
import argparse
from multiprocessing import Pool
from tqdm import tqdm, trange
import h5py
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy import constants
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
# Project imports
import utils
import sensutils
import models


BASE_PATH = os.path.split(os.path.abspath(__file__))[0]


def parse_cmdl_args():
    """Parse cmdl args to pass to main."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add arguments
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the MC/data input file.")
    parser.add_argument("--json-path", type=str, required=False,
                        help="Path to the post-processing json file.")
    parser.add_argument("--peak-shape-path", type=str, required=True,
                        help="Path to the peak_shape_data.npz file.")
    parser.add_argument("--injection-file", type=str, required=False,
                        help="Path to the injections file.")
    parser.add_argument("--dname", type=str, required=True, help="Name of the PSD dataset.")
    parser.add_argument("--dname-freq", type=str, required=True,
                        help="Name of the frequency dataset.")
    parser.add_argument("--n-processes", type=int, default=4, help="Number of processes to use.")
    parser.add_argument("--pruning", type=int, default=1,
                        help="Pruning level. Default is 1 (None).")
    parser.add_argument("--max-chi", type=int, default=10,
                        help="Maximum chi^2 deviation to skew norm fit in a chunk.")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--isMC", action="store_true")
    parser.add_argument("--regenerate-data-file", action="store_true")
    return vars(parser.parse_args())


def log_likelihood(params, Y, bkg, peak_norm, peak_shape, model_args):
    """Likelihood of finding dark matter in the data."""
    # Actual likelihood calculation
    mu_DM = params
    try:
        residuals = Y - np.log(np.exp(bkg) + peak_norm*peak_shape*mu_DM)
    except Exception as err:
        raise err
    log_lkl = sensutils.logpdf_skewnorm(
        residuals, model_args[..., 0], model_args[..., 1], model_args[..., 2])

    # Add infinity protection - breaks norm but ok if empirical tests
    mask = np.isinf(log_lkl)
    if np.sum(mask):
        row_mins = np.min(np.where(mask, np.inf, log_lkl), axis=1)
        log_lkl[mask] = np.take(row_mins, np.where(mask)[0])

    return np.sum(np.sum(log_lkl, axis=1))  # Sum over f, then over ifo


class PeakShape(np.ndarray):
    """Hold peak shape (varies over freq.)"""

    def __new__(cls, f, path, dtype=float, buffer=None, offset=0, strides=None, order=None):
        peak_arrays = np.load(path)
        shape = (peak_arrays["data"].shape[1],)
        frequency_bounds = peak_arrays["bounds"]

        obj = super(PeakShape, cls).__new__(cls, shape, dtype, buffer, offset, strides, order)
        obj.f = np.array([f])
        obj.frequency_bounds = frequency_bounds
        obj.peak_shapes = peak_arrays["data"]
        obj._update_array()
        return obj

    def _update_array(self):
        """Update the array based on the value of f."""
        condition = np.where(self.f > self.frequency_bounds)[0]
        idx = 0 if condition.size == 0 else condition[-1] + 1
        np.copyto(self, self.peak_shapes[idx, :])

    def update_freq(self, f):
        """Update the frequency value (and as such the peak shape array if necessary)."""
        self.f = np.array([f])
        self._update_array()


class DMFinder:
    """Hold DM-finding related options."""

    def __init__(self, data_path=None, json_path=None, dname=None,
                 isMC=False, peak_shape_path=None, dname_freq=None,
                 regenerate_data_file=True, max_chi=10, **kwargs):
        """Initialise necessarily shared variables and prep data."""
        # Init variables
        self.kwargs = kwargs
        self.max_chi = max_chi
        self.rho_local = 0.4 / (constants.hbar / constants.e
                                * 1e-7 * constants.c)**3  # GeV/cm^3 to GeV^4

        # Generate peak shape
        self.peak_shape = PeakShape(0, peak_shape_path)
        self.len_peak = len(self.peak_shape)

        # Read data & TF info
        self.transfer_functions, self.f_A_star = self.get_f_A_star()
        if isMC or not regenerate_data_file:
            self.dfile = h5py.File(data_path, "r")
        else:
            self.data_info = self.get_info_from_data_path(data_path, json_path)
            self.dfile = self.dfile_from_data(data_path, dname, dname_freq)
        self.dset = self.dfile[dname]
        self.freqs = self.dfile[dname_freq]

    def __del__(self):
        """Close HDF properly."""
        if hasattr(self, "data_info"):
            for dfile, _, _ in self.data_info:
                dfile.close()
        if hasattr(self, "dfile") and self.dfile:
            self.dfile.close()

    def parse_ifo(self, i):
        """Parse the ifo from the dset attr at index i."""
        return self.dset.attrs[str(i)].split("_")[-1]

    def get_info_from_data_path(self, data_path, json_path):
        """Get all possible info from result files in data_path and json_path."""
        data_paths = sorted(list(glob.glob(os.path.join(data_path, "result*"))))
        data_info = []
        for _data_path in tqdm(data_paths, desc="Opening data HDFs", leave=False):
            _data_info = []
            _data_info.append(h5py.File(sensutils.get_corrected_path(_data_path), "r"))
            ifo = sensutils.parse_ifo(_data_path)
            _data_info.append(ifo)
            df_key = "splines_" + sensutils.get_df_key(_data_path)
            _data_info.append(sensutils.get_results(df_key, json_path))
            assert _data_info[-1] is not None
            # _data_info.append(df_key)
            data_info.append(_data_info)

        return data_info

    def dfile_from_data(self, data_path, dname, dname_freq):
        """Create dfile by combining available data through data_info."""
        dfile = h5py.File(os.path.join(data_path, "tmp.h5"), "w")
        # Get PSD/freq shape
        # Using min because the difference in the frequencies between the different files
        # should only be a single bin anyway (last bin)
        psd_length = min([len(inf[0][dname_freq]) for inf in self.data_info])
        dset = dfile.create_dataset(dname, (psd_length, len(self.data_info)),
                                    dtype=np.float64)
        # Get frequencies
        for inf in self.data_info:
            if len(inf[0][dname_freq]) == psd_length:
                dfile.create_dataset(dname_freq, data=inf[0][dname_freq][:])
                break
        # Metadata
        bkg = dfile.create_dataset("bkg", (psd_length, len(self.data_info)),
                                   dtype=np.float64)
        model_args = dfile.create_dataset("model_args", (psd_length, len(self.data_info), 3),
                                          dtype=np.float64)

        # Fill data dset
        df_columns = ["fmin", "fmax", "x_knots", "y_knots",
                      "alpha_skew", "loc_skew", "sigma_skew", "chi_sqr"]
        for i, (hdf_path, ifo, df) in enumerate(
                tqdm(self.data_info, desc="Fill dset.", leave=False)):
            dset[:, i] = hdf_path["logPSD"][:psd_length]
            dset.attrs[str(i)] = ifo

            for _, row in df.iterrows():
                fmin, fmax, x_knots, y_knots, alpha_skew, loc_skew, sigma_skew, chi_sqr\
                    = row[df_columns]
                if chi_sqr >= self.max_chi:
                    continue

                # Get frequency indices
                frequencies = hdf_path["frequency"]
                idx_min = sensutils.binary_search(frequencies, fmin)
                idx_max = sensutils.binary_search(frequencies, fmax)

                # Get backgound model
                spline = models.model_xy_spline(
                    np.concatenate([x_knots, y_knots]), extrapolate=True)(
                        np.log(frequencies[idx_min:idx_max+1]))

                # Write to disk
                bkg[idx_min:idx_max+1, i] = spline
                for j, val in enumerate([alpha_skew, loc_skew, sigma_skew]):
                    model_args[idx_min:idx_max+1, i, j] = val

        # Clean dataset (loop to avoid excessive memory usage)
        for j in trange(dset.shape[1], desc="Cleaning"):
            dset[dset[:, j] == 0, j] = np.nan

        return dfile

    def get_f_A_star(self):
        """Get dicts of transfer function info."""
        tf_dir = os.path.join(BASE_PATH, "data", "transfer_functions")
        transfer_functions = {}
        transfer_functions["H1"] = pd.read_csv(os.path.join(tf_dir, "Amp_Cal_LHO.txt"),
                                            delimiter="\t")
        transfer_functions["L1"] = pd.read_csv(os.path.join(tf_dir, "Amp_Cal_LLO.txt"),
                                            delimiter="\t")
        f_A_star = {"H1": interp1d(transfer_functions["H1"]["Freq_o"],
                                transfer_functions["H1"]["Amp_Cal_LHO"]),
                    "L1": interp1d(transfer_functions["L1"]["Freq_Cal"],
                                transfer_functions["L1"]["amp_cal_LLO"])}
        return transfer_functions, f_A_star


def make_args(fndr, fmin, fmax, pruning=1):
    """Return a generator to q0 calculation args between fmin and fmax."""
    frequencies = fndr.freqs
    idx_min = sensutils.binary_search(frequencies, fmin)
    idx_max = 1 + sensutils.binary_search(frequencies, fmax)

    for idx in range(idx_min, idx_max, pruning):
        Y, bkg, model_args, peak_norm, ifos = [], [], [], [], []

        freq_Hz = frequencies[idx:idx + fndr.len_peak]
        if len(freq_Hz) < fndr.len_peak:
            continue  # We are too close to a valid boundary edge (or inside it!)

        for i in range(fndr.dset.shape[1]):  # For each segment
            # Get tf factor
            ifo = fndr.parse_ifo(i)
            A_star_sqr = fndr.f_A_star[ifo](freq_Hz)**2

            # Fill argument lists
            _Y = fndr.dset[idx:idx + fndr.len_peak, i]
            if len(_Y) < fndr.len_peak or any(_Y == 0) or any(np.isnan(_Y)):
                continue
            Y.append(_Y)
            bkg.append(fndr.dfile["bkg"][idx:idx + fndr.len_peak, i])
            # alpha, loc, sigma
            model_args.append(fndr.dfile["model_args"][idx:idx + fndr.len_peak, i, :])
            peak_norm.append(fndr.rho_local / (np.pi*freq_Hz**3 * A_star_sqr
                                               * (constants.e / constants.h)**2))
            ifos.append(ifo)
        if not Y:
            continue
        Y, bkg, model_args, peak_norm = list(map(np.array, [Y, bkg, model_args, peak_norm]))
        fndr.peak_shape.update_freq(freq_Hz[0])
        yield Y, bkg, model_args, peak_norm, fndr.peak_shape, freq_Hz[0], ifos


def plot_candidate(Y, bkg, mu, peak_norm, peak_shape, ifos):
    """Do a candidate-style plot for H1 and L1."""
    # residuals = Y - np.log(np.exp(bkg) + peak_norm*peak_shape*mu_DM)
    fig = plt.figure()
    gs = GridSpec(1, 2, figure=fig)
    axH1 = fig.add_subplot(gs[0, 0])  # First row, first column
    axL1 = fig.add_subplot(gs[0, 1], sharex=axH1, sharey=axH1)

    # Plot the PSDs
    for i in range(Y.shape[0]):
        ax = axH1 if ifos[i] == "H1" else axL1
        # ax.plot(Y[i, :], alpha=.5, zorder=1)
        ax.plot(np.exp(Y[i, :]), alpha=.5, zorder=1)
        ax.plot(np.exp(bkg[i, :]), alpha=.5, color="C0", zorder=0)

    # Find the first H1 and L1 peak norm terms
    peak_norm_H1 = peak_norm[ifos.index("H1"), :]
    peak_norm_L1 = peak_norm[ifos.index("L1"), :]

    # Min-max plots
    for ifo, ax, _peak_norm in zip(["L1", "H1"], [axL1, axH1], [peak_norm_L1, peak_norm_H1]):
        # Plot reconstructed peaks
        # ax.plot(np.log(_peak_norm*mu*peak_shape), color="k", zorder=2)
        ax.plot(_peak_norm*mu*peak_shape, color="k", zorder=2)
        # How to plot bkg info??

        #     # Choose PSD with the lowest median to add example signal to
        #     freqL, medL = None, np.inf
        #     freqM, medM = None, -np.inf
        #     for i in range(Y.shape[0]):
        #         if ifos[i] != ifo:
        #             continue

        #         _logPSD = Y[i, :]
        #         _med = np.median(_logPSD)
        #         if _med < medL:
        #             medL, logPSDL = _med, _logPSD
        #         if _med > medM:
        #             medM, logPSDM = _med, _logPSD

        #     # Subtract signal from most sensitive signal to see effect
        #     x = np.logspace(np.log10(test_freq*(1 + 1e-6)**(-len_peak)),
        #                     np.log10(test_freq*(1 + 1e-6)**len_peak),
        #                     len_peak)
        #     for freq, logPSD in zip([freqL, freqM], [logPSDL, logPSDM]):
        #         f_bkg = interp1d(freq, logPSD)

        #         peak_norm = fndr.rho_local / (np.pi * x**3 * fndr.f_A_star[ifo](x)**2
        #                                 * (constants.e / constants.h)**2)
        #         subtracted_psd = np.log(np.exp(f_bkg(x)) -
        #                                 peak_norm*mu*peak_shape)

        #         # Plot!
        #         ax.plot(x, subtracted_psd, zorder=2, color="k")
        #         ax.plot(x, f_bkg(x), zorder=2, color="k")
        #         ax.plot(x, np.log(peak_norm*mu*peak_shape), color="k", zorder=2)

        # Nice things
        ax.set_title(ifo + r", $\Lambda_i^{-1}$:" +
                        f" {np.sqrt(mu):.1e}")
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(linestyle="--", linewidth=1, color="grey", alpha=.33)
    axL1.set_ylabel("log(PSD)")
    axL1.set_yticklabels([])
    axH1.set_yscale("log")
    return axH1, axL1


def get_q0(Y, bkg, model_args, peak_norm, peak_shape,
           ifo, min_log10mu=-40, max_log10mu=-32, verbose=False):
    """Do actual q0 related calculations & minimisation, return zero_lkl, max_Lkl, mu_hat."""
    def log_lkl(params):
        return log_likelihood(params[0], Y, bkg, peak_norm, peak_shape, model_args)

    # Get initial guess
    test_mus = np.logspace(min_log10mu, max_log10mu, 1000)
    test_lkl = np.array([-log_lkl([mu]) for mu in test_mus])
    if not any(~np.isnan(test_lkl)):
        return np.nan, np.nan, np.nan
    mask = np.isnan(test_lkl) | np.isinf(test_lkl)
    initial_guess = test_mus[np.argmin(test_lkl[~mask])]

    # Calculate max lkl
    popt = minimize(lambda x: -log_lkl(x),
                    initial_guess,
                    bounds=[(0, None)],
                    method="Nelder-Mead",
                    tol=1e-10)
    max_lkl, zero_lkl = -popt.fun, log_lkl([0])
    # Debugging
    if verbose:  # or np.sqrt(popt.x[0]) > 5e-17:
        plt.figure()
        ax = plt.subplot(111)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(np.sqrt(test_mus), test_lkl)
        ax.axvline(np.sqrt(popt.x[0]), linestyle="--", color="r")
        q0 = -2 * (zero_lkl - max_lkl) if popt.x[0] > 0 else 0
        ax.set_title(r"$L(\hat{\mu}}$" +
                     f"): {max_lkl:.1e}, L(0): {zero_lkl:.1e}, Z: {np.sqrt(q0):.1f}")

        plot_candidate(Y, bkg, popt.x[0], peak_norm, peak_shape, ifo)
        plt.show()
    return zero_lkl, max_lkl, popt.x[0]


def process_q0(args):
    """Prep q0 calculation for parallel job."""
    Y, bkg, model_args, peak_norm, peak_shape, freqs, ifo = args
    try:
        zero_lkl, max_lkl, mu = get_q0(Y, bkg, model_args, peak_norm, peak_shape, ifo)
    except AssertionError:
        zero_lkl, max_lkl, mu = np.nan, np.nan, np.nan
    return freqs, zero_lkl, max_lkl, mu


def main(injection_file=None, n_processes=4, pruning=1,
         fmin=10, fmax=5000, verbose=False, **kwargs):
    """Coordinate q0 analysis."""
    # Get data (MC or Real)
    fndr = DMFinder(**kwargs)
    if injection_file is None:
        # Run data-like job
        args = make_args(fndr, fmin, fmax, pruning)

        results = []
        with Pool(n_processes) as pool:
            with tqdm(total=fndr.dset.shape[0]//pruning, position=0,
                      desc="q0 calc.", leave=True) as pbar:
                for result in pool.imap(process_q0, args):
                    results.append(result)
                    pbar.update(1)
    else:
        # Run injection-like job
        # Fill args for parallel lkl min
        injData = np.atleast_2d(utils.safe_loadtxt(injection_file, dtype=float))
        def full_args():
            for freq in tqdm(injData[:len(injData)//pruning, 0], desc="Prep args"):
                fmin, fmax = freq*(1 + 1e-6)**(-10), freq*(1 + 1e-6)**10
                for arg in make_args(fndr, fmin, fmax):
                    yield arg

        # Run parallel job
        results = []
        args = list(full_args())
        print("Starting q0 calculation..")
        with Pool(n_processes) as pool:
            with tqdm(total=injData.shape[0]*20//pruning, position=0,
                    desc="q0 calc.", leave=True) as pbar:
                for result in pool.map(process_q0, args):#(arg for arg in args)):
                                    #    chunksize=20):
                    results.append(result)
                    pbar.update(1)

    # Merge results
    q0_data = np.zeros((len(results), 4))
    for i, result in enumerate(results):
        q0_data[i, :] = result  # freqs, zero_lkl, max_lkl, mu
    np.save("q0_data.npy", q0_data)
    # q0_data = np.load("q0_data.npy")

    # Plot
    if not verbose:
        return
    pos_mu = q0_data[:, 3] > 0
    q0 = np.zeros(q0_data.shape[0])
    q0[pos_mu] = -2*(q0_data[:,   1][pos_mu] - q0_data[:, 2][pos_mu])

    # q0 vs frequency
    ax = plt.subplot(111)
    ax.set_xscale("log")
    idx = np.argsort(q0_data[:, 0])
    ax.plot(q0_data[idx, 0], np.sqrt(q0[idx]))
    ax.set_title("Z")
    ax.set_ylim(0, ax.get_ylim()[1])

    # hat(mu) vs frequency
    plt.figure()
    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(q0_data[idx, 0], np.sqrt(q0_data[idx, 3]), zorder=1)
    ax.axhline(1e-17, color="r", linestyle="--", zorder=2, linewidth=2)
    ax.set_title("Lambda_i^-1")

    # Injection sanity check
    if injection_file is None:
        plt.show()
        return
    plt.figure()
    post_mask = np.zeros(len(q0), dtype=bool)
    for freq in injData[:len(injData)//pruning, 0]:
        test_data = np.where(q0_data[idx, 0] > freq)[0]
        if len(test_data) == 0:
            continue
        fmin_idx = test_data[0]
        post_mask[fmin_idx:fmin_idx+10] = 1
    ax = plt.subplot(111)
    ax.set_yscale("log")
    _, bins, _ = ax.hist(np.sqrt(q0[idx][~post_mask]), 200, histtype="step",
                         color="C0", label="pre-peak")
    ax.hist(np.sqrt(q0[idx][post_mask]), bins, histtype="step", color="C1", label="post-peak")
    ax.legend(loc="upper right")
    ax.set_title("q_0 for injections sanity check")
    plt.show()


if __name__ == '__main__':
    main(**parse_cmdl_args())

# Example of running on MC injection: python DM_finder_v3.py --data-path ../sensitivity/MC.h5 --isMC --injection-file ../sensitivity/data/injections/injections_full_1.0e-17.dat --peak-shape-path peak_shape_data.npz --dname injection_1e-17 --dname-freq frequencies --pruning 8 --verbose
# Example of running on data: python DM_finder_v3.py --data-path data/tmp.h5 --json-path data/processing_results.json --peak-shape-pa
# th peak_shape_data.npz --dname PSD --dname-freq frequency --pruning 8 --verbose

# Careful, data-path must be a dict if --regenerate-data-file is used, example:
# python DM_finder_v3.py --data-path data --json-path data/processing_results.json --peak-shape-path peak_shape_data.npz --dname PSD --dname-freq frequency --pruning 8 --verbose --regenerate-data-file
