"""Pre-process PSDs by (BF-optimized) Fitting splines through segments."""
import os
import argparse
import glob
from multiprocessing import Pool
from tqdm import tqdm
import h5py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import skewnorm
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
# Project imports
from peak_finder import PeakFinder
import sensutils
import hist
import models


def parse_inputs():
    """Parse cmdl inputs and return dict."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-path-type", default="file", choices=["file", "dir"])
    parser.add_argument("--output-json-path", type=str, default="data/processing_results.json")
    parser.add_argument("--prefix", type=str, default="result",
                        help="Prefix of data files to use.")
    parser.add_argument("--plot-prefix", type=str, default="",
                        help="Prefix for plots in log/ if verbose.")
    parser.add_argument("--verbose", action="store_true")

    return vars(parser.parse_args())


def get_y_spline(x, *params):
    """Wrap spline model on x."""
    return models.model_xy_spline(params, extrapolate=True)(x)


def get_bic(x, y, n_knots, buffer=20, nbins=100, verbose=False):
    """Perform spline fit and return skew-norm BIC."""
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
    bounds += [(None, None) for y in y_knots]

    # 4. Minimize mean square error
    def lkl(params):
        y_spline = models.model_xy_spline(params, extrapolate=True)(x)
        return ((y - y_spline)**2).sum()
    popt = minimize(lkl, np.concatenate([x_knots, y_knots]),
                    method="Nelder-Mead", bounds=bounds)

    y_spline = models.model_xy_spline(popt.x, extrapolate=True)(x)
    if verbose:
        print(popt)
        plt.plot(x, y, zorder=1)
        plt.plot(x, y_spline, zorder=2)
        plt.scatter(popt.x[:n_knots], popt.x[n_knots:], color="r", zorder=3)

    # 5. Get starting parameters for skew normal
    res = y - y_spline
    alpha, mu, sigma = sensutils.get_skewnorm_p0(res)

    # 6. Fit a skew normal
    def fitFunc(_x, *params):
        return params[3]*skewnorm.pdf(_x, params[0], loc=params[1], scale=params[2])

    h0, bins = np.histogram(res, nbins)
    try:
        _popt, _ = hist.fit_hist(fitFunc, res, bins, [alpha, mu, sigma, max(h0)])
    except RuntimeError:
        return -np.inf, popt.x, [alpha, mu, sigma]

    # 7. Calculate BIC :-)
    _lkl = skewnorm.pdf(res, _popt[0], loc=_popt[1], scale=_popt[2])
    bic = _lkl.sum() - 0.5 * (2*len(x_knots) + 4) * np.log(len(x))

    if verbose:
        ax = hist.plot_func_hist(fitFunc, _popt, res, bins)
        ax.set_title(f"BIC: {bic:.1e}")
        plt.show()

    return bic, popt.x, _popt[:-1]


def process_iteration(params):
    """Wrap call to bayesian_regularized_linreg in parallel."""
    i, x, y, kwargs, verbose, plot_prefix = params
    best_fit, f_popt, distr_popt = sensutils.bayesian_regularized_linreg(x, y, **kwargs)
    if best_fit is None:
        if verbose:
            prefix = f"[{np.exp(x[0]):.3f}-{np.exp(x[-1]):.3f}] Hz"
            plt.plot(x, y)
            plt.savefig(os.path.join("log", f"{plot_prefix}fail_{prefix}.pdf"))
            plt.close()
        return i, None, None, np.inf

    # Calculate chi-sqr of distr fit (Poisson uncertainty available)
    y_spline = models.model_xy_spline(f_popt, extrapolate=True)(x)
    h0, bins = np.histogram(y - y_spline, kwargs["nbins"] if "nbins" in kwargs else 100)
    bin_centers = (bins[1:] + bins[:-1]) / 2.
    m = h0 != 0
    distr = np.sum(h0)*(bins[1] - bins[0])*skewnorm.pdf(
        bin_centers, distr_popt[0], loc=distr_popt[1], scale=distr_popt[2])
    chi_sqr = np.sum((h0[m] - distr[m])**2 / h0[m]) / (len(h0[m]) - len(distr_popt))

    if verbose:
        prefix = f"[{np.exp(x[0]):.3f}-{np.exp(x[-1]):.3f}] Hz"
        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(4, 1)
        ax, axRes = fig.add_subplot(gs[:3]), fig.add_subplot(gs[3])
        ax.plot(x, y)
        ax.plot(x, best_fit)
        ax.set_title(f"{prefix}, " + r"$\chi^2$:" + f"{chi_sqr:.1f}, k:{len(f_popt)//2}")
        ax.set_ylabel("log(PSD)")
        axRes.set_xlabel("log(Hz)")
        axRes.grid(linestyle="--", color="grey", alpha=.5)
        axRes.plot(x, y - best_fit, ".", zorder=1)
        axRes.axhline(0, color="r", zorder=2)
        # Plot knots
        k = len(f_popt) // 2
        x_knots, y_knots = f_popt[:k], f_popt[k:]
        ax.scatter(x_knots, y_knots, color="r")
        plt.savefig(os.path.join("log", f"{plot_prefix}{prefix}_spline.pdf"))
        plt.savefig(os.path.join("log", f"{plot_prefix}{prefix}_spline.png"))
        plt.close()
    return i, f_popt, distr_popt, chi_sqr


def process_hybrid(data_path, json_path, pruning=1, n_processes=1, plot_prefix="",
                   ana_fmin=10, ana_fmax=5000, segment_size=10000, verbose=False,
                   k_min=4, max_plateau=3, k_pruning=1, buffer=40, nbins=50):  # bic args
    """For now just a BIC testing area."""
    # Get the block-corrected noPeak data
    with h5py.File(sensutils.get_corrected_path(data_path)) as _f:
        peak_mask = np.array(_f["peak_mask"][()], dtype=bool)
        freq = _f["frequency"][()][~peak_mask]
        logPSD = _f["logPSD"][()][~peak_mask]
        del peak_mask

    # Get the results DataFrame
    df_key = "splines_" + sensutils.get_df_key(data_path)
    df = sensutils.get_results(df_key, json_path)
    if df is None:
        df = pd.DataFrame(columns=["fmin", "fmax", "x_knots", "y_knots",
                                   "alpha_skew", "loc_skew", "sigma_skew", "chi_sqr"])

    # Minimise using bic
    idx_start = np.where(freq >= ana_fmin)[0][0]
    idx_end = np.where(freq <= ana_fmax)[0][-1]
    positions = np.concatenate([
        np.arange(idx_start, idx_end, segment_size),
        [idx_end]])

    def make_args():
        kwargs = dict(get_bic=get_bic, f_fit=get_y_spline, k_min=k_min, max_plateau=max_plateau,
                      k_pruning=k_pruning, plot_mean=True, kernel_size=800, verbose=False,
                      buffer=buffer, nbins=nbins)
        for i, (start, end) in enumerate(zip(positions[:-1], positions[1:])):
            # Skip if the entry already exists
            if not (len(df) <= i or (len(df) > i and df.iloc[i].isnull().any())):
                continue
            kwargs["disable_tqdm"] = i != len(df) + 1
            x, y = np.log(freq[start:end:pruning]), logPSD[start:end:pruning]
            if len(x) <= 1:
                continue
            yield i, x, y, kwargs, verbose, plot_prefix

    # Parallel calculation
    with Pool(processes=n_processes, maxtasksperchild=10) as pool:
        results = []
        with tqdm(total=len(positions)-1-len(df), desc="Minimise BIC") as pbar:
            for result in pool.imap_unordered(process_iteration, make_args()):
                results.append(result)
                pbar.update(1)

    # Post-processing
    zipped_positions = list(zip(positions[:-1], positions[1:]))
    for result in tqdm(results, desc="Combine results"):
        i, f_popt, distr_popt, chi_sqr = result
        if np.isinf(chi_sqr):
            continue
        start, end = zipped_positions[i]

        assert not len(f_popt) % 2
        n_knots = len(f_popt) // 2
        df = pd.concat([df, pd.DataFrame(
            {"fmin": freq[start],
             "fmax": freq[end],
             "x_knots": [list(f_popt[:n_knots])],
             "y_knots": [list(f_popt[n_knots:])],
             "alpha_skew": distr_popt[0],
             "loc_skew": distr_popt[1],
             "sigma_skew": distr_popt[2],
             "chi_sqr": chi_sqr
             })])
        sensutils.update_results(df_key, df, json_path, orient="records")
    if verbose:
        df = sensutils.get_results(df_key, json_path)
        plt.figure()
        plt.plot(df["chi_sqr"])
        plt.title("chi_sqr/dof")
        plt.show()


def correct_blocks(data_path, json_path, whiten=True, verbose=False, **kwargs):
    """
    Apply peak finding block correction and return path to data.

    The first block correction enables peak finding,
    which enables a second iteration of whitening.
    """
    # Check if corrected data is already available, if not, generate it
    corrected_path = sensutils.get_corrected_path(data_path)
    df_name = sensutils.get_df_key(data_path)
    peak_info = sensutils.get_results(df_name, json_path)
    if os.path.exists(corrected_path) and peak_info is not None:
        return

    # Set variables
    kwargs["name"] = data_path
    pf = PeakFinder(**kwargs)
    pruning = 1000
    buffer = 500
    _SEGMENT_SIZE = int(1e4)  # Size of chunks for skew norm fits
    _CFD_ALPHA = .05  # CFD threshold for peak zoom in fits
    _CHI_LOW, _CHI_HIGH = .1, 2  # Quality cuts on fit results
    _CUT_ALPHA = .99  # How much of the whitened norm data to keep

    # Get knots from block positions
    block_positions = list(pf.block_position_gen())
    x_knots = np.array([np.log(pf.freq[pos])
                        for pos in block_positions[:-1]] + [np.log(pf.freq[-1])])
    y_knots = np.array([np.log(np.median(pf.psd[pos:pos+buffer]))
                        for pos in block_positions[:-1]] + [np.log(pf.psd[-1])])

    # Fit slopes on top of spline
    popt, _ = pf.line_only_fit(x_knots, y_knots, block_positions,
                               np.zeros(len(pf.psd), dtype=bool),
                               pruning=pruning, verbose=verbose)

    # Combined fit using previous result as initial guess
    bf, Y_model = pf.combine_spline_slope_smart(x_knots, y_knots, block_positions, popt,
                                                np.zeros(len(pf.psd), dtype=bool),
                                                pruning=pruning, verbose=verbose)

    # Fit chunks with skew normals
    residuals = np.log(pf.psd) - Y_model
    del Y_model
    popt_chunks, _, chis_chunks = pf.fit_frequency_blocks(
        residuals, _CFD_ALPHA, _SEGMENT_SIZE, _CHI_LOW, _CHI_HIGH
    )

    # Whiten fitted residuals to find peaks
    peak_mask, _ = pf.get_peak_mask_from_residuals(
        residuals, np.zeros(len(pf.psd), dtype=bool), popt_chunks, chis_chunks,
        _SEGMENT_SIZE, _CHI_LOW, _CHI_HIGH, _CUT_ALPHA, verbose=False
    )

    # Final fit on peak-less data (we are fitting the background after all)
    bf, Y_model = pf.combine_spline_slope_smart(x_knots, y_knots, block_positions,
                                                bf[3+len(x_knots):], peak_mask,
                                                pruning=pruning, verbose=verbose)

    # Second peak-finding iteration on cleaned data
    residuals = np.log(pf.psd) - Y_model
    popt_chunks, _, chis_chunks = pf.fit_frequency_blocks(
        residuals[~peak_mask], _CFD_ALPHA, _SEGMENT_SIZE, _CHI_LOW, _CHI_HIGH,
    )
    del Y_model

    # Whiten fitted residuals to find peaks
    _, peak_info = pf.get_peak_mask_from_residuals(
        residuals, peak_mask, popt_chunks, chis_chunks,
        _SEGMENT_SIZE, _CHI_LOW, _CHI_HIGH, _CUT_ALPHA,
        pruning=100, verbose=verbose
    )
    del residuals

    # Save results
    Y_corrected = np.log(pf.psd) - models.model_segment_slope(
        bf, np.log(pf.freq), block_positions, shift=3+len(x_knots))
    with h5py.File(corrected_path, "w") as _f:
        _f.create_dataset("logPSD", data=Y_corrected, dtype="float64")
        _f.create_dataset("frequency", data=pf.freq, dtype="float64")
        _f.create_dataset("peak_mask", data=peak_mask, dtype="bool")

    df_peak = pd.DataFrame(peak_info, columns=["start_idx", "width", "max"])
    sensutils.update_results(df_name, df_peak, json_path)


def main(data_path, output_json_path, plot_prefix="", verbose=False):
    """Organise analysis."""
    kwargs = {"epsilon": 0.1,
              "fmin": 10,
              "fmax": 8192,
              "fs": 16384,
              "resolution": 1e-6
              }
    # tmp - remove fit data
    # corrected_path = sensutils.get_corrected_path(data_path)
    # df_key = "splines_" + sensutils.get_df_key(data_path)
    # sensutils.del_df_from_json(df_key, json_path)

    # Correct for block structure and find peaks based on that model
    correct_blocks(data_path, output_json_path, verbose=verbose, **kwargs)

    # Fit a bayesian regularized spline model using a skew normal likelihood
    # Used on block-corrected noPeak data
    process_hybrid(data_path, output_json_path, ana_fmin=10, ana_fmax=5000,
                   segment_size=1000, k_min=4, k_pruning=1, max_plateau=3,
                   buffer=40, nbins=75, pruning=3,
                   plot_prefix=plot_prefix, verbose=verbose, n_processes=10)


if __name__ == '__main__':
    cmdl_kwargs = parse_inputs()
    path = cmdl_kwargs.pop("data_path")
    path_type = cmdl_kwargs.pop("data_path_type")
    prefix = cmdl_kwargs.pop("prefix")

    if path_type == "file":
        main(path, **cmdl_kwargs)
    else:
        for path in glob.glob(os.path.join(path, f"{prefix}*")):
            main(path, **cmdl_kwargs)
