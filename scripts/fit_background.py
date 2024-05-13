"""Spline fit log-log background in frequency chunks."""
import os
import argparse
from multiprocessing import Pool
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize
# Project imports
from LPSDIO import LPSDOutput, LPSDJSONIO
import models  # only to be called in BkgModel
import stats
import hist


def parse_args():
    """Define cmdl. arg. parser and return vars."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to lpsd output file or folder of files.")
    parser.add_argument("--data-prefix", type=str, default="",
                        help="If data-path is a dir, only consider files starting with prefix.")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to output json file.")
    parser.add_argument("--ana-fmin", type=float, default=10.,
                        help="Minimum frequency to consider.")
    parser.add_argument("--ana-fmax", type=float, default=5000.,
                        help="Maximum frequency to consider.")
    parser.add_argument("--segment-size", type=int, default=1000,
                        help="Size of frequency segment in which to fit the splines in bins.")
    parser.add_argument("--k-min", type=int, default=4,
                        help="Minimum number of spline knots (>= 4)")
    parser.add_argument("--k_pruning", type=int, default=1,
                        help="How many k steps to skip in opt.")
    parser.add_argument("--k_plateau", type=int, default=3,
                        help="How long to wait for improvement before search ends.")
    parser.add_argument("--buffer", type=int, default=50,
                        help="Add frequency bins to add around segment (regularization)")
    parser.add_argument("--bin-scaling", type=float, default=.15,
                        help="Bin scaling factor for the skew-normal hist fit.")
    parser.add_argument("--pruning", type=int, default=1,
                        help="Prune data for speed-up.")
    parser.add_argument("--n-processes", type=int, default=1,
                        help="Number of processes to use for multiprocessing of main loop.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot-path", type=str, default=None,
                        help="If verbose, path to dir in which to save plots.")
    parser.add_argument("--kernel-size", type=int, default=30,
                        help="If verbose, smoothing kernel size in BIC plot.")
    parser.add_argument("--cfd-fraction", type=float, default=.1,
                        help="Regularize fits given outliers.")

    return vars(parser.parse_args())


def apply_cfd(h0, bins, _cfd):
    """Apply rising and falling edge constant fraction discriminator."""
    peak_index = np.argmax(h0)
    rising_idx_col = np.where(h0[:peak_index] >= h0[peak_index] * _cfd)[0]
    if not list(rising_idx_col):
        rising_idx = 0
    else:
        rising_idx = rising_idx_col[0]

    falling_idx_col = np.where(h0[peak_index:] < h0[peak_index] * _cfd)[0]
    if not list(falling_idx_col) or falling_idx_col[0] > len(h0) -2:
        falling_idx = len(h0) - 2
    else:
        falling_idx = falling_idx_col[0] + peak_index

    return np.linspace(bins[rising_idx], bins[falling_idx+1], len(bins))


def bkg_model(x, knots):
    """Actual spline bkg-model function."""
    return models.model_xy_spline(knots, extrapolate=True)(x)


def calc_bic(x, y, n_knots, bin_scaling=.15, buffer=40, cfd_fraction=.1):
    """
    Fit the background while optimising BIC.

    Space x_knots logarithmically, starting params: y_knot on data median
    As an approximation, get spline params by minimising mean square
    Then fit background hist to get skew norm params
    Use sum-knowledge to calculate skew-norm likelihood
    """
    # 1. Choose x_knots
    x_knots = np.linspace(min(x), max(x), n_knots)
    # 1.1 Choose y_knots based on y data
    y_knots = np.array([np.median(y[max(0, np.where(x >= knot)[0][0] - buffer):
        np.where(x >= knot)[0][0] + buffer]) for knot in x_knots])

    # 2. Bound x in non-overlapping blocks
    bounds = [(x_knots[0], (x_knots[1] + x_knots[0]) / 2)]
    epsilon = 0.001 * (bounds[0][1] - bounds[0][0])
    bounds += [(0.5 * (x_knots[i] + x_knots[i-1]) + epsilon,
                0.5 * (x_knots[i] + x_knots[i+1]) - epsilon)
                for i in range(1, len(x_knots)-1)]
    bounds += [(0.5 * (x_knots[-2] + x_knots[-1]), x_knots[-1])]
    # 3. Add y bounds
    bounds += [(None, None) for _ in y_knots]

    # 4. Minimize mean square error to find spline params
    def rms(params):
        y_model = bkg_model(x, params)
        return ((y - y_model)**2).sum()
    popt = minimize(rms, np.concatenate([x_knots, y_knots]),
                    method="Nelder-Mead", bounds=bounds)
    # assert popt.success
    y_model = bkg_model(x, popt.x)

    # 5. Get starting parameters for skew normal
    res = y - y_model
    alpha, mu, sigma = stats.get_skewnorm_p0(res)

    # 6. Fit a skew normal
    def fitFunc(_x, *params):
        return stats.pdf_skewnorm(_x, params[3], alpha=params[2],
                                    loc=params[0], scale=params[1])

    a, b = np.quantile(res, [0.05, 0.95])
    bin_width = bin_scaling * np.std(res[(res > a) & (res < b)])  # EMPIRICAL

    bins = np.arange(min(res), max(res)+bin_width, bin_width)
    h0, bins = np.histogram(res, bins)
    # 6.1 Use CFD to regularize in the face of outliers
    bins = apply_cfd(h0, bins, cfd_fraction)
    bins = np.arange(bins[0], bins[-1]+bin_width, bin_width)
    cfd_mask = (res >= bins[0]) & (res <= bins[-1])

    try:
        _popt, _ = hist.fit_hist(fitFunc, res, bins, [mu, sigma, alpha, max(h0)])
    except RuntimeError:
        return -np.inf, popt.x, [alpha, mu, sigma]

    # 7. Use all previous info as starting params for full lkl fit.
    def log_lkl(params):
        residuals = y[cfd_mask] - bkg_model(x[cfd_mask], params[3:])
        # lkl = stats.logpdf_skewnorm(residuals, *params[:3])  # would be faster
        # Safely calculate log pdf values
        pdf = stats.pdf_skewnorm(residuals, 1, *params[:3])
        lkl = np.full(pdf.shape, -np.inf, dtype=np.float64)
        mask = pdf == 0
        lkl[~mask] = np.log(pdf[~mask])

        # Protect against NaNs
        isnan = np.isnan(lkl)
        if isnan.sum() > 0:
            try:
                lkl[isnan] = min(lkl[~isnan])
            except ValueError:
                return np.inf
        return -lkl.sum()

    p0 = np.concatenate([_popt[:3], popt.x])
    bounds = [[v - 0.1*abs(v), v + 0.1*abs(v)] for v in p0[:3]] + bounds
    total_popt = minimize(log_lkl, p0, method="Nelder-Mead", bounds=bounds)

    # 7. Calculate BIC :-)
    y_model = bkg_model(x[cfd_mask], total_popt.x[3:])
    res = y[cfd_mask] - y_model
    _lkl = stats.pdf_skewnorm(res, 1, *total_popt.x[:3])  # loc, scale, alpha
    bic = _lkl.sum() - 0.5 * (2*len(x_knots) + 4) * np.log(len(x))

    # assert total_popt.success
    return bic, total_popt.x[3:], total_popt.x[:3]


def bayesian_regularized_linreg(
    x, y, get_bic, f_fit, k_min=4, k_pruning=1, k_plateau=2,
        disable_tqdm=False, **bic_kwargs):
    """Fit polynomials for different degrees and select the one with the highest BIC."""
    # Loop over k
    orders, bics = [], []
    best_bic, best_f_popt, best_distr_popt = -np.inf, None, None

    # Loop until bic stops improving
    i, no_improvement_count = 0, 0
    order = k_min
    pbar = tqdm(disable=disable_tqdm, desc="BIC..", position=1, leave=False)
    while True:
        # Perform the fit
        bic, f_popt, distr_popt = get_bic(x, y, order, **bic_kwargs)
        bics.append(bic)
        orders.append(order)

        if bic > best_bic:
            best_bic = bic
            best_f_popt, best_distr_popt = f_popt, distr_popt
            no_improvement_count = 0  # Reset the counter if there's an improvement
        else:
            no_improvement_count += 1  # Increment the counter if no improvement

        # Check for early stopping
        if no_improvement_count > k_plateau:
            break

        # Loop conditions
        pbar.update(1)
        i += 1
        order += k_pruning
    pbar.close()

    if best_f_popt is None:
        return None, None, None
    return f_fit(x, best_f_popt), best_f_popt, best_distr_popt


def process_iteration(args):
    """Wrap BIC-minimising bkg. fit."""
    i, x, y, kwargs = args
    kernel_size, plot_mean, verbose, plot_path = list(map(
        kwargs.pop, ["kernel_size", "plot_mean", "verbose", "plot_path"]))
    if verbose:
        prefix = f"{np.exp(x[0]):.3f}_{np.exp(x[-1]):.3f}_Hz"
        try:
            plot_path = os.path.join(plot_path, f"{prefix}.png")
        except TypeError as err:
            print("[ERROR] You need to specify a plot_path if verbose is True.")
            raise err

    y_model, func_popt, distr_popt = bayesian_regularized_linreg(
        x, y, calc_bic, bkg_model, **kwargs)
    # Catch fit errors
    if y_model is None:
        if verbose:
            plt.figure()
            plt.plot(x, y)
            plt.savefig(plot_path)
            plt.close()
        return i, None, None, np.inf

    # Calculate chi-sqr of distr. fit through Poisson
    res = y - y_model
    a, b = np.quantile(res, [0.05, 0.95])  # Base binning on std, protect from outliers
    bin_width = kwargs["bin_scaling"] * np.std(res[(res > a) & (res < b)])

    # Apply CFD
    bins = np.arange(min(res), max(res)+bin_width, bin_width)
    h0, bins = np.histogram(res, bins)
    bins = apply_cfd(h0, bins, kwargs["cfd_fraction"])
    h0, bins = np.histogram(res,
                            np.arange(bins[0], bins[-1]+bin_width, bin_width))

    bin_centers, bin_width = (bins[1:] + bins[:-1]) / 2., bins[1] - bins[0]
    pos_h0 = h0 != 0
    residuals_fit = stats.pdf_skewnorm(bin_centers, np.sum(h0) * bin_width, *distr_popt)
    chi_sqr = np.sum((h0[pos_h0] - residuals_fit[pos_h0])**2 / h0[pos_h0])

    # Save figure
    if verbose:
        gs = GridSpec(1, 4)
        fig = plt.figure(figsize=(8, 4.5))
        axData, axHist = fig.add_subplot(gs[:3]), fig.add_subplot(gs[-1])

        axData.plot(x, y, ".", label="Data", zorder=1)
        k = len(func_popt) // 2
        axData.plot(x, y_model, linewidth=2., zorder=3, label=f"Best fit (k: {k})")
        axData.scatter(func_popt[:k], func_popt[k:], color="r", zorder=4)
        if plot_mean:
            axData.plot(x, stats.kde_smoothing(y, kernel_size), label="kde", zorder=2)
        axData.legend(loc="best")
        axData.set_ylabel("logPSD")

        axHist.hist(res, bins, histtype="step")
        axHist.plot(bin_centers, residuals_fit)
        axHist.set_xlabel("Residuals (logPSD)")
        axHist.set_title(r"Skew-normal $\chi^2/dof$: " + f"{chi_sqr / (np.sum(pos_h0) - 3):.1f}")

        plt.savefig(plot_path)
        plt.close()

    return i, func_popt, distr_popt, chi_sqr / (np.sum(pos_h0) - 3)


def fit_background_in_file(data_path, output_path, n_processes=1, buffer=50,
                           pruning=1, segment_size=10000, ana_fmin=10, ana_fmax=5000,
                           **kwargs):
    """
    Fit a background model to power spectral density from a LPSD or LPSDOutput file.

    Parameters:
        data_path (str): Path to the input data file.
        output_path (str): Path to the (L)PSD, .h5 or .lpsd.
        n_processes (int): Number of parallel processes to use.
        buffer (int): To adjust data segment boundaries.
        pruning (int): For data reduction.
        segment_size (int): Size of each data segment for fitting.
        ana_fmin (float): Minimum frequency.
        ana_fmax (float): Maximum frequency.
        **kwargs: Additional keyword arguments for the fitting function.

    Raises:
        ValueError: If the data file does not exist.
    """
    if not os.path.exists(data_path):
        print(f"[ERROR] The file '{data_path}' does not exist, aborting..")
        raise ValueError
    data = LPSDOutput(data_path)

    # Start parallel BIC-based calculation
    idx_start = np.where(data.freq >= ana_fmin)[0][0]
    idx_end = np.where(data.freq <= ana_fmax)[0][-1]
    positions = np.concatenate([
        np.arange(idx_start, idx_end, segment_size),
        [idx_end]])
    def make_args():
        kwargs.update({"plot_mean": True})
        for i, (start, end) in enumerate(zip(positions[:-1], positions[1:])):
            start, end = max(0, start - buffer), end + buffer
            x, y = np.log(data.freq[start:end:pruning]), data.logPSD[start:end:pruning]
            if len(x) <= 4:
                continue
            kwargs["disable_tqdm"] = i > 0
            yield i, x, y, kwargs

    with Pool(processes=n_processes, maxtasksperchild=10) as pool:
        results = []
        with tqdm(total=len(positions)-1, desc="Bkg. Fit Freq. progress") as pbar:
            for result in pool.imap_unordered(process_iteration, make_args()):
                results.append(result)
                pbar.update(1)

    # Combine results
    df = pd.DataFrame(columns=["fmin", "fmax", "x_knots", "y_knots",
                               "alpha_skew", "loc_skew", "sigma_skew", "chi_sqr"])
    zipped_positions = list(zip(positions[:-1], positions[1:]))
    for result in tqdm(results, desc="Combine results"):
        i, f_popt, distr_popt, chi = result
        start, end = zipped_positions[i]

        assert not len(f_popt) % 2
        n_knots = len(f_popt) // 2
        df = pd.concat([df, pd.DataFrame(
            {"fmin": data.freq[start],
             "fmax": data.freq[end],
             "x_knots": [list(f_popt[:n_knots])],
             "y_knots": [list(f_popt[n_knots:])],
             "alpha_skew": distr_popt[2],
             "loc_skew": distr_popt[0],
             "sigma_skew": distr_popt[1],
             "chi_sqr": chi
             })])

    mngr = LPSDJSONIO(output_path)
    mngr.update_file(data_path, df)


if __name__ == '__main__':
    _kwargs = parse_args()
    _data_path = _kwargs.pop("data_path")
    _output_path = _kwargs.pop("output_path")
    _data_prefix = _kwargs.pop("data_prefix")

    if os.path.isdir(_data_path):
        for file_path in glob.glob(os.path.join(_data_path, f"{_data_prefix}*")):
            fit_background_in_file(file_path, _output_path, **_kwargs)
    elif os.path.exists(_data_path):
        fit_background_in_file(_data_path, _output_path, **_kwargs)
    else:
        raise IOError(f"Invalid path: '{_data_path}'.")
