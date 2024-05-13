import os
import argparse
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from scipy import constants
from scipy.optimize import minimize
import h5py
# Project imports
from LPSDIO import LPSDDataGroup, LPSDJSONIO, get_A_star
from fit_background import bkg_model
import stats


def parse_args():
    """Define cmdl. arg. parser and return vars."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to lpsd output file or folder of files.")
    parser.add_argument("--data-prefix", type=str, default="",
                        help="If data-path is a dir, only consider files starting with prefix.")
    parser.add_argument("--peak-shape-file", type=str, required=True,
                        help="Path to the peak shape numpy file.")
    parser.add_argument("--bkg-info-path", type=str, required=True,
                        help="Output data from fit_background.py.")
    parser.add_argument("--tf-path", type=str, required=True,
                        help="Path to directory holding transfer functions.")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to file (ending in .h5 or .hdf5) to store the results.")
    parser.add_argument("--iteration", type=int, default=0,
                        help="Condor iteration number")
    parser.add_argument("--batch-size", type=int, default=-1,
                        help="If positive, number of frequencies to run over.")
    parser.add_argument("--ana-fmin", type=float, default=10,
                        help="Minimum analysis frequency.")
    parser.add_argument("--ana-fmax", type=float, default=5000,
                        help="Maximum analysis frequency.")
    parser.add_argument("--max-chi-sqr", type=float, default=10.,
                        help="Maximum chi^2/dof value for skew-norm fits.")
    parser.add_argument("--n-processes", type=int, default=1,
                        help="If > 1, use multiprocessing.")
    parser.add_argument("--min-log10mu", type=int, default=-40,
                        help="Min of the log10mu values to investigate.")
    parser.add_argument("--max-log10mu", type=int, default=-32,
                        help="Max of the log10mu values to investigate.")
    return vars(parser.parse_args())


def find_leading_trailing_invalid(arr):
    """Find the indices of the leading and trailing groups of invalid numbers in an array."""
    # Create a boolean mask where True indicates either NaN or Inf
    mask = np.isnan(arr) | np.isinf(arr)
    if not np.sum(~mask):
        return 0, 0

    # Find the last index of the leading group of invalid numbers
    if mask[0]:
        for i, val in enumerate(mask):
            if not val:
                first_idx = i - 1
                break
    else:
        first_idx = 0

    # Find the first index of the trailing group of invalid numbers
    if mask[-1]:
        for i, val in enumerate(mask[::-1]):
            if not val:
                last_idx = len(arr) - 1 - (i - 1)
                break
    else:
        last_idx = len(arr) - 1

    return first_idx + 1, last_idx - 1


def get_valid_mu_ranges(Y, bkg, peak_norm, model_args,
                        start=-40, end=-30, n=100):
    """Investigate where mu is valid for an easier later analysis."""
    mu = np.logspace(start, end, n)
    output = []
    def get_mu_ranges(test_mus):
        valid_mu_ranges = np.zeros((len(Y), 2))
        for i, logPSD in enumerate(Y):
            residuals = logPSD - np.log(np.exp(bkg[i]) + peak_norm[i]*test_mus)
            log_lkl_values = np.log(stats.pdf_skewnorm(residuals, *model_args[i]))
            start, end = find_leading_trailing_invalid(log_lkl_values)

            if not np.isnan(log_lkl_values[0]) and start:
                start -= 1
            assert not np.sum(np.isnan(log_lkl_values[start:end]))

            valid_mu_ranges[i, :] = test_mus[start], test_mus[end]

        return valid_mu_ranges

    # Positive values
    valid_mu_ranges_pos = get_mu_ranges(mu)
    valid_mu_range_pos = np.max(valid_mu_ranges_pos[:, 0]), np.min(valid_mu_ranges_pos[:, 1])
    if valid_mu_range_pos[1] - valid_mu_range_pos[0] > 0:
        output.append(valid_mu_range_pos)

    # Negative values
    valid_mu_ranges_neg= get_mu_ranges(-mu[::-1])
    valid_mu_range_neg = np.max(valid_mu_ranges_neg[:, 0]), np.min(valid_mu_ranges_neg[:, 1])
    if valid_mu_range_neg[1] - valid_mu_range_neg[0] > 0:
        output.append(valid_mu_range_neg)

    return output


def calc_max_lkl(Y, bkg, model_args, peak_norm, peak_shape,
                 min_log10mu=-40, max_log10mu=-32, n_seed_lkl=1000):
    """Calculate maximum likelihood information."""
    # Question: do I calculate sigma here as well? Might as well no?
    def log_lkl_shape(params):
        return stats.log_likelihood(params[0], Y, bkg, peak_norm, peak_shape, model_args)

    # Get initial guess for mu, based on peak_shape
    test_mus = np.logspace(min_log10mu, max_log10mu, n_seed_lkl)
    test_lkl = np.array([-log_lkl_shape([mu]) for mu in test_mus])
    mask = np.isnan(test_lkl) | np.isinf(test_lkl)
    if not any(~mask):
        return np.nan, np.nan, np.nan, np.nan
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

    # Calculate uncertainty on mu
    valid_mu_ranges = get_valid_mu_ranges(Y, bkg, peak_norm, model_args,
                                          start=min_log10mu, end=max_log10mu, n=1000)
    max_dist = min(mu - valid_mu_ranges[0][0], valid_mu_ranges[0][1] - mu)\
        if mu > 0 else min(mu - valid_mu_ranges[1][0], valid_mu_ranges[1][1] - mu)
    try:
        # Derive uncertainty from Fisher matrix
        sigma = stats.sigma_at_point(lambda x: log_lkl_shape([x]),
                                     mu,
                                     initial_dx=min(max_dist/2., abs(mu)),
                                     tolerance=1e-4)
        assert not np.isnan(sigma)
    except (ValueError, AssertionError):
        return np.nan, np.nan, np.nan, np.nan
    # TODO: Could also calculate zero-side of two-sided exact logL sigma
    return zero_lkl, max_lkl, mu, sigma


def maximise_likelihood(args):
    """Multiprocessing wrapper for maximum likelihood estimation."""
    Y, bkg, model_args, peak_norm, peak_shape, test_freq, kwargs = args
    zero_lkl, max_lkl, mu, sigma = calc_max_lkl(Y, bkg, model_args, peak_norm, peak_shape, **kwargs)
    return test_freq, zero_lkl, max_lkl, mu, sigma


def main(data_path=None, peak_shape_file=None, output_path=None, bkg_info_path=None,
         tf_path=None, data_prefix="", iteration=0, ana_fmin=10, ana_fmax=5000, batch_size=-1,
         max_chi_sqr=10, n_processes=1, min_log10mu=-40, max_log10mu=-32):
    """Multiprocess maximum likelihood calculation"""
    # Get LPSD data
    data = LPSDDataGroup(data_path, data_prefix)
    if not os.path.exists(data.buffer_path):
        data.save_data()
    peak_shape = np.load(peak_shape_file)
    peak_size = len(peak_shape)
    f_A_star = get_A_star(tf_path)
    # TODO: Calibration

    # Get background info
    bkg_data = LPSDJSONIO(bkg_info_path)
    bkg_info = [bkg_data.get_df(k) for k in bkg_data.data]
    df_columns = ["x_knots", "y_knots", "alpha_skew", "loc_skew", "sigma_skew", "chi_sqr"]

    # Find frequency subset
    idx_start = np.where(data.freq >= ana_fmin)[0][0]
    idx_end = np.where(data.freq >= ana_fmax)[0][0]
    if batch_size < 0:
        frequencies = data.freq[idx_start:idx_end]
    else:
        assert idx_start + iteration*batch_size < idx_end
        frequencies = data.freq[idx_start + iteration*batch_size:
                                idx_start + (iteration+1)*batch_size]

    lkl_kwargs = {"min_log10mu": min_log10mu, "max_log10mu": max_log10mu}
    def make_args():
        for i, test_freq in enumerate(frequencies):
            freq_idx = idx_start + int(iteration * batch_size) + i
            # Skip this entry if test_freq is out of bounds
            if freq_idx + peak_size > len(data.freq):
                continue

            Y = data.logPSD[:, freq_idx:freq_idx + peak_size]
            freq_Hz = data.freq[freq_idx:freq_idx + peak_size]

            bkg, model_args, peak_norm = [], [], []
            for j, df in enumerate(bkg_info):
                mask = (df['fmin'] <= test_freq) & (df['fmax'] >= test_freq)
                x_knots, y_knots, alpha_skew, loc_skew, sigma_skew, chi_sqr\
                    = df[mask][df_columns].iloc[0]
                # Skip this entry if the fit is bad
                if chi_sqr >= max_chi_sqr:
                    continue

                _bkg = bkg_model(np.log(freq_Hz), np.concatenate([x_knots, y_knots]))
                _model_args = [alpha_skew, loc_skew, sigma_skew]
                ifo = data.metadata["ifo"][j]
                _beta = stats.RHO_LOCAL / (np.pi*freq_Hz**3 * f_A_star[ifo](freq_Hz)**2
                                           * (constants.e / constants.h)**2)

                # Fill arg arrays
                bkg.append(_bkg)
                model_args.append(_model_args)
                peak_norm.append(_beta)
            Y, bkg, peak_norm, model_args = list(map(np.array, [Y, bkg, peak_norm, model_args]))
            yield Y, bkg, model_args, peak_norm, peak_shape, test_freq, lkl_kwargs

    # Run maximum likelihood calculation in parallel
    results = []
    with Pool(n_processes, maxtasksperchild=100) as pool,\
            tqdm(total=len(frequencies), position=0, desc="Max lkl calc.", leave=True) as pbar:
        for result in pool.imap(maximise_likelihood, make_args()):
            results.append(result)
            pbar.update(1)

    # Merge results
    lkl_data = np.zeros((len(results), 5))
    for i, result in enumerate(results):
        lkl_data[i, :] = result  # freq, max_lkl, zero_lkl, mu_hat, sigma
    lkl_data = lkl_data[~np.isnan(lkl_data[:, 0]), :]  # Clean
    lkl_data = lkl_data[np.argsort(lkl_data[:, 0]), :]  # Sort

    _locals = locals()
    vars_to_save = ['data_path', 'peak_shape_file', 'output_path', 'bkg_info_path', 'tf_path',
                    'data_prefix', 'iteration', 'ana_fmin', 'ana_fmax', 'batch_size',
                    'max_chi_sqr', 'n_processes', 'min_log10mu', 'max_log10mu']
    kwargs = {name: str(_locals[name]) for name in vars_to_save}
    with h5py.File(output_path, "w") as _file:
        _file.create_dataset("lkl_data", data=lkl_data)
        _file["lkl_data"].attrs.update(kwargs)


if __name__ == '__main__':
    main(**parse_args())
