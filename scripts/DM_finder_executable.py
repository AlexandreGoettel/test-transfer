"""Run q0 calculation - condor version."""
import os
import glob
import argparse
from multiprocessing import Pool
from tqdm import tqdm
import h5py
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy import constants
from scipy.stats import skewnorm
import pandas as pd
import numpy as np
# Project imports
import sensutils
import models
import utils


BASE_PATH = os.path.split(os.path.abspath(__file__))[0]


def parse_cmdl_args():
    """Parse cmdl args."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n-frequencies", type=int, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--ana-fmin", type=float, required=True)
    parser.add_argument("--ana-fmax", type=float, required=True)
    parser.add_argument("--Jdes", type=int, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--json-path", type=str, required=True)
    parser.add_argument("--peak-shape-path", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True, help="Where to store results.")
    parser.add_argument("--data-prefix", type=str, default="result")

    parser.add_argument("--isMC", action="store_true")
    parser.add_argument("--injection-path", type=str, default=None)
    parser.add_argument("--injection-peak-shape-path", type=str, default=None)

    parser.add_argument("--max-chi", type=float, default=10)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--n-processes", type=int, default=1)

    return vars(parser.parse_args())


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

    def __init__(self, data_path=None, json_path=None, peak_shape_path=None,
                 injection_path=None, injection_peak_shape_path=None,
                 max_chi=10, data_prefix="result", **_):
        """Initialise necessarily shared variables and prep data."""
        # Init variables
        self.max_chi = max_chi
        self.injection_path = injection_path
        self.rho_local = 0.4 / (constants.hbar / constants.e
                                * 1e-7 * constants.c)**3  # GeV/cm^3 to GeV^4

        # Generate peak shape
        self.peak_shape = None if peak_shape_path is None else PeakShape(0, peak_shape_path)
        self.injection_peak_shape = self.peak_shape if injection_peak_shape_path is None else\
            PeakShape(0, injection_peak_shape_path)
        self.len_peak = 0 if self.peak_shape is None else len(self.peak_shape)

        # Read data & TF info
        self.transfer_functions, self.f_A_star = self.get_f_A_star()
        self.data_info = self.get_info_from_data_path(data_path, json_path,
                                                      prefix=data_prefix)

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

    def get_info_from_data_path(self, data_path, json_path, prefix="result"):
        """Get all possible info from result files in data_path and json_path."""
        data_paths = sorted(list(glob.glob(os.path.join(data_path, f"{prefix}*"))))
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


def log_likelihood(params, Y, bkg, peak_norm, peak_shape, model_args):
    """Likelihood of finding dark matter in the data."""
    # Actual likelihood calculation
    assert Y.shape[0] == bkg.shape[0] and Y.shape[1] == peak_shape.shape[0]
    mu_DM = params
    try:
        residuals = Y - np.log(np.exp(bkg) + peak_norm*peak_shape*mu_DM)
    except Exception as err:
        raise err
    log_lkl = sensutils.logpdf_skewnorm(
        residuals, model_args[..., 0], model_args[..., 1], model_args[..., 2])
    assert log_lkl.shape == Y.shape

    # Add infinity protection - breaks norm but ok if empirical tests
    mask = np.isinf(log_lkl)
    if np.sum(mask):
        row_mins = np.min(np.where(mask, np.inf, log_lkl), axis=1)
        log_lkl[mask] = np.take(row_mins, np.where(mask)[0])

    return np.sum(np.sum(log_lkl, axis=1))


def get_q0(Y, bkg, model_args, peak_norm, peak_shape,
           min_log10mu=-40, max_log10mu=-32):
    """Do actual q0 related calculations & minimisation, return zero_lkl, max_Lkl, mu_hat."""
    def log_lkl_shape(params):
        return log_likelihood(params[0], Y, bkg, peak_norm, peak_shape, model_args)

    # Get initial guess for mu, based on peak_shape
    test_mus = np.logspace(min_log10mu, max_log10mu, 1000)
    test_lkl = np.array([-log_lkl_shape([mu]) for mu in test_mus])
    mask = np.isnan(test_lkl) | np.isinf(test_lkl)
    if not any(~mask):
        return 0, 0, 0
    initial_guess = test_mus[np.argmin(test_lkl[~mask])]

    # Calculate max lkl
    popt_peak_shape = minimize(
        lambda x: -log_lkl_shape(x),
        [initial_guess],
        bounds=[(0, None)],
        method="Nelder-Mead",
        tol=1e-10)

    # Get max & zero lkl
    zero_lkl = log_lkl_shape([0])
    max_lkl = -popt_peak_shape.fun
    mu = popt_peak_shape.x[0]

    return zero_lkl, max_lkl, mu


def process_q0(args):
    """Wrap calculate_q0 for multiprocessing."""
    Y, bkg, model_args, peak_norm, peak_shape, fmin = args
    try:
        zero_lkl, max_lkl, mu = get_q0(Y, bkg, model_args, peak_norm, peak_shape)
    except AssertionError:
        zero_lkl, max_lkl, mu = np.nan, np.nan, np.nan
    return fmin, zero_lkl, max_lkl, mu


def make_args(fndr, ana_fmin, ana_fmax, Jdes, n_freqs, iteration=0, isMC=False):
    """Generator to prep. parallel q0 calculation."""
    # Get process variables
    freqs = np.logspace(np.log10(ana_fmin), np.log10(ana_fmax),
                        Jdes)[iteration*n_freqs:(iteration+1)*n_freqs]

    # Fill args
    df_columns = ["x_knots", "y_knots", "alpha_skew", "loc_skew", "sigma_skew", "chi_sqr"]
    for test_freq in freqs:
        Y, bkg, model_args, peak_norm = [], [], [], []
        for hdf_path, ifo, df in fndr.data_info:
            mask = (df['fmin'] <= test_freq) & (df['fmax'] >= test_freq)
            # Skip this entry if test_freq is out of bounds
            if np.sum(np.array(mask, dtype=int)) == 0:
                continue
            x_knots, y_knots, alpha_skew, loc_skew, sigma_skew, chi_sqr\
                = df[mask][df_columns].iloc[0]
            # Skip this entry if fit is bad
            if chi_sqr >= fndr.max_chi_sqr:
                continue

            # Get data location
            _frequencies = hdf_path["frequency"]
            idx = sensutils.binary_search(_frequencies, test_freq)
            freq_Hz = _frequencies[idx:idx + fndr.len_peak]

            # Skip this entry if freq. bound is reached
            if idx+fndr.len_peak >= len(_frequencies):
                continue

            # Get data values
            _beta = fndr.rho_local / (np.pi*freq_Hz**3 * fndr.f_A_star[ifo](freq_Hz)**2
                                      * (constants.e / constants.h)**2)
            _bkg = models.model_xy_spline(np.concatenate([x_knots, y_knots]),
                                          extrapolate=True)(np.log(freq_Hz))
            _model_args = [alpha_skew, loc_skew, sigma_skew]

            if isMC:
                residuals = skewnorm.rvs(alpha_skew, loc=loc_skew,
                                         scale=sigma_skew, size=fndr.len_peak)
                _Y = _bkg + residuals
            else:
                _Y = hdf_path["logPSD"][idx:idx + fndr.len_peak]

            # Fill args
            Y.append(_Y)
            bkg.append(_bkg)
            model_args.append(_model_args)
            peak_norm.append(_beta)

        if not Y:
            continue
        fndr.peak_shape.update_freq(test_freq)
        Y, bkg, peak_norm, model_args = list(map(np.array, [Y, bkg, peak_norm, model_args]))
        yield Y, bkg, model_args, peak_norm, fndr.peak_shape[:], test_freq


def main(n_frequencies=35000, iteration=0, Jdes=None, ana_fmin=10, ana_fmax=5000, isMC=False,
         outdir=None, prefix="", n_processes=1, **kwargs):
    """Run q0 calculation for selected frequencies."""
    fndr = DMFinder(**kwargs)
    args = make_args(fndr, ana_fmin, ana_fmax, Jdes, n_frequencies, iteration=iteration, isMC=isMC)

    results = []
    with Pool(n_processes, maxtasksperchild=100) as pool,\
            tqdm(total=n_frequencies, position=0,
                 desc="q0 calc.", leave=True) as pbar:
        for result in pool.imap(process_q0, args):
            results.append(result)
            pbar.update(1)

    # Merge results
    q0_data = np.zeros((len(results), 4))
    for i, result in enumerate(results):
        q0_data[i, :] = result  # freqs, zero_lkl, max_lkl, mu
    np.save(os.path.join(outdir, f"{prefix}_{iteration}_result.npy"), q0_data)


if __name__ == '__main__':
    main(**parse_cmdl_args())
