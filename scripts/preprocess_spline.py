"""Pre-process PSDs by (BF-optimized) Fitting splines through segments."""
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import skewnorm
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
# Custom imports
import ultranest
from ultranest.plot import cornerplot
# Project imports
from peak_finder import PeakFinder
import sensutils
import hist
import models


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
        vectorized=True, ndraw_min=nlive//2,
        resume="overwrite", log_dir=f"log/{prefix}_"
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


def process(filename, df_path, ana_fmin=10, ana_fmax=8192, k_init=5,
            nlive=128, ev_threshold=.5, verbose=False, **kwargs):
    """Perform pre-processing on given LPSD output file."""
    kwargs["name"] = filename
    segment_size = kwargs.pop("segment_size")
    pf = PeakFinder(**kwargs)
    try:
        df = pd.read_json(df_path)
    except ValueError:
        df = pd.DataFrame(columns=["frequencies", "x_knots", "y_knots", "alpha", "sigma_sqr"])

    # 0. Separate in segments
    idx_start = np.where(pf.freq >= ana_fmin)[0][0]
    idx_end = np.where(pf.freq <= ana_fmax)[0][-1]
    positions = np.concatenate([
        np.arange(idx_start, idx_end, segment_size),
        [idx_end]])
    for i, (start, end) in enumerate(tqdm(zip(
            positions[:-1], positions[1:]), position=0, leave=True,
                                          desc="Segments", total=len(positions)-1)):
        # Skip if the entry already exists
        if not (len(df) <= i or (len(df) > i and df.iloc[i].isnull().all())):
            continue

        x, y = np.log(pf.freq[start:end]), np.log(pf.psd[start:end])
        prefix = f"{np.exp(x[0]):.1f}_{np.exp(x[-1]):.1f}_Hz"

        # do-while logic until the evidence stops improving
        ev_df = pd.DataFrame(columns=["k", "ev", "bf"])
        for k in range(k_init - 1, k_init + 2):
            x_knots = np.linspace(x[0], x[-1], k)
            evidence, bf = perform_single_fit(
                x, y, x_knots,
                nlive=nlive, prefix=prefix, dlogz=ev_threshold)
            ev_df = pd.concat([ev_df, pd.DataFrame(
                {"k": k, "ev": evidence, "bf": [list(bf)]})])

        if ev_df['ev'].max() == ev_df["ev"].iloc[-1]:
            while ev_df["ev"].iloc[-1] > ev_df["ev"].iloc[-2] + ev_threshold:
                k += 1
                x_knots = np.linspace(x[0], x[-1], k)
                new_ev, new_bf = perform_single_fit(
                    x, y, x_knots, nlive=nlive, prefix=prefix, dlogz=ev_threshold)
                ev_df = pd.concat([ev_df, pd.DataFrame(
                    {"k": k, "ev": new_ev, "bf": [list(new_bf)]})])

        idx = np.argmax(ev_df["ev"])
        k, bf = ev_df["k"].iloc[idx], ev_df["bf"].iloc[idx]
        x_knots = np.linspace(x[0], x[-1], k)
        if verbose:
            # Plot spline
            y_model = CubicSpline(x_knots, bf[2:])(x)
            plt.figure()
            plt.plot(x, y, color="C0", zorder=1)
            plt.plot(x, y_model, color="C1", zorder=2)
            plt.scatter(x_knots, bf[2:], color="r", zorder=3)
            plt.savefig(os.path.join("log", f"{prefix}_spline.png"))

            # Plot residuals
            res = y - y_model
            h0, bins = np.histogram(res, np.linspace(min(res), max(res), 50))
            bin_centers = (bins[1:] + bins[:-1]) / 2.
            alpha_skew, sigma_skew = bf[0], np.sqrt(np.abs(bf[1]))
            mode = -sensutils.get_mode_skew(0, sigma_skew, alpha_skew)

            ax = hist.plot_hist(res, bins, density=True)
            ax.plot(bin_centers, skewnorm.pdf(bin_centers, alpha_skew, scale=sigma_skew, loc=mode))
            m = h0 == 0
            chi_sqr = np.sum((h0[~m]/np.sum(h0) - skewnorm.pdf(
                bin_centers[~m], alpha_skew, scale=sigma_skew, loc=mode))**2
                             / h0[~m] * np.sum(h0)) / (len(h0[~m] - 2))
            ax.set_title(r"$\chi^2: " + f"{chi_sqr:.1f}")
            plt.savefig(os.path.join("log", f"{prefix}_residuals.png"))

            # Plot evidence over iterations
            plt.figure()
            plt.plot(ev_df["k"], ev_df["ev"], marker="o")
            plt.title("Evidence vs iteration")
            plt.savefig(os.path.join("log", f"{prefix}_evidence.png"))
            plt.close()

        # Checkpointing
        df = pd.concat([df, pd.DataFrame({"frequencies": [[start, end]],
                                          "x_knots": [list(x_knots)],
                                          "y_knots": [list(bf[2:])],
                                          "alpha": bf[0],
                                          "sigma_sqr": bf[1]})])
        df.to_json(df_path, orient="records")


def correct_blocks(path, **kwargs):
    """
    Apply peak finding block correction and return path to data.

    The first block correction enables peak finding,
    which enables a second iteration of whitening.
    """
    # Set variables
    kwargs["name"] = path
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
        _SEGMENT_SIZE, _CHI_LOW, _CHI_HIGH, _CUT_ALPHA, verbose=True
    )

    # Final fit on peak-less data (we are fitting the background after all)
    bf, Y_model = pf.combine_spline_slope_smart(x_knots, y_knots, block_positions,
                                                bf[3+len(x_knots):], peak_mask,
                                                pruning=pruning, verbose=False)
    residuals = np.log(pf.psd) - Y_model
    del Y_model
    popt_chunks, _, chis_chunks = pf.fit_frequency_blocks(
        residuals[~peak_mask], _CFD_ALPHA, _SEGMENT_SIZE, _CHI_LOW, _CHI_HIGH,
    )

    # Whiten fitted residuals to find peaks
    peak_mask, peak_info = pf.get_peak_mask_from_residuals(
        residuals, peak_mask, popt_chunks, chis_chunks,
        _SEGMENT_SIZE, _CHI_LOW, _CHI_HIGH, _CUT_ALPHA,
        pruning=100, verbose=True
    )


def main():
    """Organise analysis."""
    # TODO: rel. path
    path = "data/result_epsilon10_1243393026_1243509654_H1.txt"
    kwargs = {"epsilon": 0.1,
              "fmin": 10,
              "fmax": 8192,
              "fs": 16384,
              "resolution": 1e-6
              }
    df_path = os.path.join("preprocessing",
                           os.path.split(path)[-1].split(".")[0] + ".json")
    # Check if blocks have been corrected for
    corrected_path = correct_blocks(path, **kwargs)
    # process(path, df_path, segment_size=10000, ana_fmin=10, ana_fmax=5000,
    #         k_init=4, ev_threshold=1, nlive=64, verbose=True, **kwargs)


if __name__ == '__main__':
    main()
