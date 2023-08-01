"""Pre-process PSDs by (BF-optimized) Fitting splines through segments."""
from tqdm import tqdm
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


def perform_single_fit(x, y, x_knots, prefix, nlive=128, verbose=False):
    """Perform single Bayesian optimisation, return evidence."""
    # 1. Set initial guess along straight line
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
                             epsilon=.04, n_sigma=3)

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
        viz_callback=False
    )
    bf = results["maximum_likelihood"]["point"]
    evidence = results["logz"]

    # Plot and return
    if verbose:
        # Plot spline
        y_model = CubicSpline(x_knots, bf[2:])(x)
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, y_model)

        # Plot residuals
        res = y - y_model
        bins = np.linspace(min(res), max(res), 50)
        bin_centers = (bins[1:] + bins[:-1]) / 2.
        alpha_skew, sigma_skew = bf[0], np.sqrt(np.abs(bf[1]))
        mode = -sensutils.get_mode_skew(0, sigma_skew, alpha_skew)

        ax = hist.plot_hist(res, bins, density=True)
        ax.plot(bin_centers, skewnorm.pdf(bin_centers, alpha_skew, scale=sigma_skew, loc=mode))

        # Plot corner
        cornerplot(sampler.results)
        plt.show()

    return evidence


def process(filename, ana_fmin=10, ana_fmax=8192, k_init=5, verbose=False, **kwargs):
    """Perform pre-processing on given LPSD output file."""
    kwargs["name"] = filename
    segment_size = kwargs.pop("segment_size")
    pf = PeakFinder(**kwargs)

    # 0. Separate in segments
    idx_start = np.where(pf.freq >= ana_fmin)[0][0]
    idx_end = np.where(pf.freq <= ana_fmax)[0][-1]
    positions = np.concatenate([
        np.arange(idx_start, idx_end, segment_size),
        [idx_end]])
    for start, end in tqdm(zip(positions[:-1], positions[1:]),
                           position=0, leave=True, desc="Segments"):
        x, y = np.log(pf.freq[start:end]), np.log(pf.psd[start:end])
        x_knots = np.linspace(x[0], x[-1], k_init)

        perform_single_fit(x, y, x_knots,
                           nlive=128, prefix=str(start), verbose=verbose)
        return


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
    process(path, segment_size=10000, ana_fmin=10, ana_fmax=5000, verbose=True, **kwargs)


if __name__ == '__main__':
    main()
