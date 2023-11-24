"""q0-based significance search."""
import os
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


BASE_PATH = os.path.split(os.path.abspath(__file__))[0]


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


class DMFinder:
    """Hold DM-finding related options."""

    def __init__(self, data_path=None, dname=None, **kwargs):
        """Initialise necessarily shared variables and prep data."""
        # Init variables
        self.dname = dname
        self.kwargs = kwargs
        self.rho_local = 0.4 / (constants.hbar / constants.e
                                * 1e-7 * constants.c)**3  # GeV/cm^3 to GeV^4

        # Generate peak shape
        # TODO: REPLACE WITH INJECTION-FIT TEMPLATE
        self.peak_shape = np.load("../sensitivity/peak_shape.npy")
        self.peak_shape /= np.sum(self.peak_shape)
        self.len_peak = len(self.peak_shape)

        # Read data & TF info
        self.transfer_functions, self.f_A_star = self.get_f_A_star()
        self.dfile = h5py.File(data_path, "r")
        self.dset = self.dfile[dname]
        # self.dset, self.fset = self.read_MC_data() if isMC else self.read_real_data()

    def __del__(self):
        """Close HDF properly."""
        if hasattr(self, "data_info"):
            for [dfile, _, _] in self.data_info:
                dfile.close()
        if hasattr(self, "dfile") and self.dfile:
            self.dfile.close()

    def parse_ifo(self, i):
        """Parse the ifo from the dset attr at index i."""
        return self.dset.attrs[str(i)].split("_")[-1]

    def read_real_data(self):
        """Read from a real data repo."""
        pass  # TODO

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


def make_args(fndr, fmin, fmax):
    """Return a generator to q0 calculation args between fmin and fmax."""
    frequencies = fndr.dfile["frequencies"][:]
    idx_min = sensutils.binary_search(frequencies, fmin)
    idx_max = 1 + sensutils.binary_search(frequencies, fmax)

    for idx in range(idx_min, idx_max):
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
    # axH1.set_ylim(max(np.log(peak_norm*mu*peak_shape)) - 5,
    #                 axH1.get_ylim()[1])
    # axL1.set_ylim(axH1.get_ylim())
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
    if verbose or np.sqrt(popt.x[0]) > 5e-17:
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


def main(injection_file=None, n_processes=4, **kwargs):
    """Coordinate q0 analysis."""
    # Get data (MC or Real)
    fndr = DMFinder(**kwargs)
    pruning = 1
    # TODO: If no injection, input fmin, fmax from exterior for condor prep
    # fmin, fmax = None, None
    # For now, Else:

    # Fill args for parallel lkl min
    injData = np.atleast_2d(utils.safe_loadtxt(injection_file, dtype=float))
    def full_args():
        for freq in tqdm(injData[:len(injData)//pruning, 0], desc="Prep args"):
            fmin, fmax = freq*(1 + 1e-6)**(-10), freq*(1 + 1e-6)**10
            for arg in make_args(fndr, fmin, fmax):
                yield arg

    # Run parallel job
    # results = []
    # args = list(full_args())
    # print("Starting q0 calculation..")
    # with Pool(n_processes) as pool:
    #     with tqdm(total=injData.shape[0]*20//pruning, position=0,
    #               desc="q0 calc.", leave=True) as pbar:
    #         for result in pool.imap(process_q0, (arg for arg in args)):
    #                             #    chunksize=20):
    #             results.append(result)
    #             pbar.update(1)

    # # Merge results
    # q0_data = np.zeros((len(results), 4))
    # for i, result in enumerate(results):
    #     q0_data[i, :] = result  # freqs, zero_lkl, max_lkl, mu
    # np.save("q0_data.npy", q0_data)
    q0_data = np.load("q0_data.npy")

    # Plot
    pos_mu = q0_data[:, 3] > 0
    q0 = np.zeros(q0_data.shape[0])
    q0[pos_mu] = -2*(q0_data[:,   1][pos_mu] - q0_data[:, 2][pos_mu])

    ax = plt.subplot(111)
    ax.set_xscale("log")
    idx = np.argsort(q0_data[:, 0])
    ax.plot(q0_data[idx, 0], np.sqrt(q0[idx]))
    # for freq in injData[:len(injData)//pruning, 0]:
    #     ax.axvline(freq, linestyle="--", linewidth=.5, color="r")
    ax.set_title("Z")
    ax.set_ylim(0, ax.get_ylim()[1])

    plt.figure()
    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(q0_data[idx, 0], np.sqrt(q0_data[idx, 3]), zorder=1)
    ax.axhline(1e-17, color="r", linestyle="--", zorder=2, linewidth=2)
    ax.set_title("Lambda_i^-1")

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
    _, bins, _ = ax.hist(np.sqrt(q0[idx][~post_mask]), 200, histtype="step", color="C0", label="pre-peak")
    ax.hist(np.sqrt(q0[idx][post_mask]), bins, histtype="step", color="C1", label="post-peak")
    ax.legend(loc="upper right")
    ax.set_title("q_0 for injections sanity check")
    plt.show()


if __name__ == '__main__':
    main(data_path="../sensitivity/MC.h5",
         injection_file="../sensitivity/data/injections/injections_full_1.0e-19.dat",
         dname="injection_1e-17",
         n_processes=10
         )
