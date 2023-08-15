"""Pre-process PSDs by (BF-optimized) Fitting splines through segments."""
import os
from tqdm import tqdm
import h5py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import skewnorm, poisson
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.signal import convolve
# Custom imports
import ultranest
from ultranest.plot import cornerplot
# Project imports
from peak_finder import PeakFinder
import sensutils
import hist
import models


def get_corrected_path(data_path):
    """Generate path to store block-corrected results."""
    return os.path.join(
        os.path.split(data_path)[0],
        "block_corrected_" + ".".join(os.path.split(data_path)[1].split(".")[:-1]) + ".hdf5"
    )


def get_df_key(data_path):
    """Get a unique df name based on the LPSD output file name."""
    return os.path.split(data_path)[-1].split(".")[-2]


def prior_uniform(cube, initial, sigma, epsilon=.05, n_sigma=3):
    """Define uniform prior around initial*(1 +- epsilon)."""
    param = cube[np.newaxis, :].copy() if cube.ndim == 1 else cube.copy()
    shift = len(sigma)

    # Spline params
    lo, hi = initial[shift:] * (1. - epsilon), initial[shift:] * (1. + epsilon)
    param[:, shift:] = lo + param[:, shift:] * (hi - lo)

    # Distribution params
    lo, hi = initial[:shift] - n_sigma*sigma, initial[:shift] + n_sigma*sigma
    param[:, :shift] = lo + param[:, :shift] * (hi - lo)
    return param


def model_spline(cube, x_data, x_knots, shift=2):
    """Vectorize spline model."""
    output = np.zeros((cube.shape[0], len(x_data)))
    for i in range(cube.shape[0]):
        output[i, :] = CubicSpline(x_knots, cube[i, shift:], extrapolate=False)(x_data)
    return output


def likelihood_spline_skew(cube, x, y_data, x_knots, shift=2):
    """Define skew-normal likelihood function around spline."""
    y_model = model_spline(cube, x, x_knots, shift=shift)
    alpha, sigma = cube[:, 0], np.sqrt(np.abs(cube[:, 1]))
    mode = sensutils.get_mode_skew(np.zeros_like(alpha), sigma, alpha)
    lkl = skewnorm.logpdf((y_data - y_model).T,
                          alpha,
                          scale=sigma,
                          loc=-mode)
    return np.sum(lkl, axis=0)


def perform_single_fit(x, y, x_knots, prefix,
                       nlive=128, dlogz=.5, pruning=10, verbose=False):
    """Perform single Bayesian optimisation, return evidence."""
    # 1. Set initial guess along straight line
    x, y = x[::pruning], y[::pruning]
    a, b = np.polyfit(x, y, 1)

    # 2. Find initial parameters for the skew norm shape
    def fitFunc(x, A, mu, alpha, sigma):
        return A*skewnorm.pdf(x, alpha, scale=np.sqrt(sigma), loc=mu)

    res = y - CubicSpline(x_knots, a*x_knots+b, extrapolate=False)(x)
    bins = np.linspace(min(res), max(res), 100)
    popt, pcov = hist.fit_hist(fitFunc, res, bins, p0=[len(x), 0, 0, 1])
    skew_uncertainty = np.sqrt(np.diag(pcov)[2:])

    # 3. Find total initial parameters using global minimisation
    initial_guess = np.concatenate(
        [[popt[2], popt[3]], a*x_knots + b +
         sensutils.get_mode_skew(popt[1], np.sqrt(popt[3]), popt[2])])
    # bounds = [(-10, 10), (.1, None)]
    # bounds += [(min(y), max(y)) for x in x_knots]
    def lkl(params):
        out = likelihood_spline_skew(params[None, :], x, y, x_knots, shift=2)
        return -out
    popt = minimize(lkl, x0=initial_guess, method="Powell")
    assert popt.success

    # 3. Start Bayesian optimisation procedure around optimal point
    def prior(cube, *_):
        return prior_uniform(cube, popt.x, skew_uncertainty,
                             epsilon=.03, n_sigma=3)

    def likelihood(cube, *_):
        return likelihood_spline_skew(cube, x, y, x_knots, shift=2)

    parameters = [r"$\alpha$", r"$\sigma$"] + [f"$k_{i}$" for i in range(len(x_knots))]
    sampler = ultranest.ReactiveNestedSampler(
        parameters, likelihood, prior,
        vectorized=True, ndraw_min=nlive//2
        # resume="overwrite", log_dir=f"log/{prefix}_"
    )
    results = sampler.run(
        min_num_live_points=nlive,
        dlogz=dlogz,
        viz_callback=False
    )
    bf = results["maximum_likelihood"]["point"]
    evidence = results["logz"]

    if verbose:
        cornerplot(sampler.results)
        plt.show()

    return evidence, bf


def get_y_spline(x, *params):
    """Apply spline model on x."""
    assert not len(params) % 2
    n_knots = len(params) // 2
    params = np.array(params)

    return CubicSpline(params[:n_knots], params[n_knots:], extrapolate=True)(x)


def get_bic(x, y, n_knots, buffer=20, nbins=100, verbose=False):
    """Perform spline fit and return skew-norm BIC."""
    # 1. Choose x_knots
    x_knots = np.linspace(min(x), max(x), n_knots)
    # 1.1 Choose y_knots based on y data
    y_knots = np.array([np.median(y[max(0, np.where(x >= knot)[0][0] - buffer):
        np.where(x >= knot)[0][0] + buffer]) for knot in x_knots])

    # 2. Bound x in non-overlapping blocks
    bounds = [(x_knots[0], (x_knots[1] + x_knots[0]) / 2)]
    bounds += [(0.5 * (x_knots[i] + x_knots[i-1]),
                0.5 * (x_knots[i] + x_knots[i+1]))
                for i in range(1, len(x_knots)-1)]
    bounds += [(0.5 * (x_knots[-2] + x_knots[-1]), x_knots[-1])]
    # 3. Add y bounds
    bounds += [(None, None) for y in y_knots]

    # 4. Minimize mean square error
    def lkl(params):
        y_spline = get_y_spline(x, *params)
        return ((y - y_spline)**2).sum()
    popt = minimize(lkl, np.concatenate([x_knots, y_knots]),
                    method="Nelder-Mead", bounds=bounds)

    y_spline = get_y_spline(x, *popt.x)
    if verbose:
        print(popt)
        plt.plot(x, y, zorder=1)
        plt.plot(x, y_spline, zorder=2)
        plt.scatter(popt.x[:n_knots], popt.x[n_knots:], color="r", zorder=3)

    # 5. Get starting parameters for skew normal
    res = y - y_spline
    alpha, mu, sigma = sensutils.get_skewnorm_p0(res)

    # 6. Fit a skew normal
    def fitFunc(x, *params):
        return params[3]*skewnorm.pdf(x, params[0], loc=params[1], scale=params[2])

    h0, bins = np.histogram(res, nbins)
    _popt, _ = hist.fit_hist(fitFunc, res, bins, [alpha, mu, sigma, max(h0)])

    if verbose:
        hist.plot_func_hist(fitFunc, _popt, res, bins)
        plt.show()

    # 7. Calculate BIC :-)
    lkl = skewnorm.pdf(res, _popt[0], loc=_popt[1], scale=_popt[2])
    bic = lkl.sum() - 0.5 * (2*len(x_knots) + 4) * np.log(len(x))

    return bic, popt.x, _popt[:-1]


def process_hybrid(filename, json_path, pruning=1,
                   ana_fmin=10, ana_fmax=5000, segment_size=10000,
                   k_min=4, k_max=20, buffer=40, nbins=50,  # bic args
                   verbose=False, **kwargs):
    """For now just a BIC testing area."""
    kwargs["name"] = filename
    pf = PeakFinder(**kwargs)
    # Get the results DataFrame
    df_key = "splines_" + get_df_key(filename)
    df = sensutils.get_results(df_key, json_path)
    if df is None:
        df = pd.DataFrame(columns=["frequencies", "x_knots", "y_knots",
                                   "alpha_skew", "loc_skew", "sigma_skew"])

    # Minimise using bic
    idx_start = np.where(pf.freq >= ana_fmin)[0][0]
    idx_end = np.where(pf.freq <= ana_fmax)[0][-1]
    positions = np.concatenate([
        np.arange(idx_start, idx_end, segment_size),
        [idx_end]])
    for start, end in tqdm(
        zip(positions[:-1], positions[1:]), position=0, leave=True,
            desc="Segments", total=len(positions)-1):
        x, y = np.log(pf.freq[start:end:pruning]), np.log(pf.psd[start:end:pruning])
        best_fit, f_popt, distr_popt = sensutils.bayesian_regularized_linreg(
            x, y, get_bic=get_bic, f_fit=get_y_spline, k_min=k_min, k_max=k_max,
            plot_mean=True, kernel_size=800, verbose=False,
            buffer=buffer, nbins=nbins)  # bic_kwargs

        if verbose:
            prefix = f"[{x[0]:.2f}-{x[-1]:.2f}] Hz"
            plt.figure()
            plt.plot(x, y)
            plt.plot(x, best_fit)
            plt.title(prefix)
            plt.xlabel("log(Hz)")
            plt.ylabel("log(PSD)")
            plt.savefig(os.path.join("log", f"{prefix}_spline.png"))
            plt.close()

        # Checkpointing
        assert not len(f_popt) % 2
        n_knots = len(f_popt) // 2
        df = pd.concat([df, pd.DataFrame({"frequencies": [[start, end]],
                                          "x_knots": [list(f_popt[:n_knots])],
                                          "y_knots": [list(f_popt[n_knots:])],
                                          "alpha_skew": distr_popt[0],
                                          "loc_skew": distr_popt[1],
                                          "sigma_skew": distr_popt[2]})])
        sensutils.update_results(df_key, df, json_path, orient="records")


def correct_blocks(data_path, json_path, verbose=False, **kwargs):
    """
    Apply peak finding block correction and return path to data.

    The first block correction enables peak finding,
    which enables a second iteration of whitening.
    """
    # Check if corrected data is already available, if not, generate it
    corrected_path, df_name = get_corrected_path(data_path), get_df_key(data_path)
    peak_info = sensutils.get_results(df_name, json_path)
    if os.path.exists(corrected_path) and peak_info is not None:
        return

    # Set variables
    kwargs["name"] = data_path
    pf = PeakFinder(**kwargs)
    pruning = 10000
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
                               pruning=pruning, verbose=False)
    # Combined fit using previous result as initial guess
    bf, Y_model = pf.combine_spline_slope_smart(x_knots, y_knots, block_positions, popt,
                                                np.zeros(len(pf.psd), dtype=bool),
                                                pruning=pruning, verbose=False)

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
                                                pruning=pruning, verbose=False)

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
    del residuals, peak_mask

    # Save results
    Y_corrected = np.log(pf.psd) - models.model_segment_slope(
        bf, np.log(pf.freq), block_positions, shift=3+len(x_knots))
    with h5py.File(corrected_path, "w") as _f:
        _f.create_dataset("logPSD", data=Y_corrected, dtype="float64")
        _f.create_dataset("frequency", data=pf.freq, dtype="float64")

    df_peak = pd.DataFrame(peak_info, columns=["start_idx", "width", "max"])
    sensutils.update_results(df_name, df_peak, json_path)


def main():
    """Organise analysis."""
    # TODO: rel. paths
    json_path = "data/processing_results.json"
    data_path = "data/result_epsilon10_1243393026_1243509654_H1.txt"
    kwargs = {"epsilon": 0.1,
              "fmin": 10,
              "fmax": 8192,
              "fs": 16384,
              "resolution": 1e-6
              }
    # Correct for block structure and find peaks based on that model
    correct_blocks(data_path, json_path, **kwargs)

    # Fit a bayesian regularized spline model using a skew normal likelihood
    process_hybrid(data_path, json_path, ana_fmin=10, ana_fmax=5000,
                   segment_size=10000, k_min=4, k_max=20,
                   buffer=40, nbins=50, pruning=2,
                   verbose=True, **kwargs)


if __name__ == '__main__':
    main()
