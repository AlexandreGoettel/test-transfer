"""Fit background (L)PSD with spline + segment lines."""
import json
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import ultranest
import ultranest.stepsampler
import pymultinest
# Project imports
from multinest_reader import MultiNestAnalyser
import utils
import models


def block_position_gen(fmin, fmax, resolution, fs, epsilon):
    """Generator for block positions."""
    # Get LPSD variables
    g = np.log(fmax) - np.log(fmin)
    Jdes = int(np.floor(1 + g / np.log(1. + resolution)))

    # Initial state
    j0 = 0
    j = utils.j(j0, Jdes, g, epsilon, fmin, fmax, fs, resolution)
    yield 0  # Always start at 0
    yield int(j)

    # Loop over frequencies
    while True:
        j0 = j
        j = utils.j(j0, Jdes, g, epsilon, fmin, fmax, fs, resolution)
        if j >= Jdes - 1:
            yield Jdes - 1
            return
        yield int(j)


def model_spline(cube, x_knots, shift=1):
    """Cubic spline fit to be used in log-log space."""
    y_knots = cube[shift:len(x_knots)+shift]
    return CubicSpline(x_knots, y_knots, extrapolate=False)


def simple_spline_alt():
    """Placeholder alternative to launching multinest."""
    x_knots = np.array([2.30258509, 3.04941017, 3.79623525,
                        8.65059825, 8.95399594, 8.97733423, 9.02401079])
    y_knots = np.array([-91.365, -97.333, -101.16,
                        -101.98, -105.33, -107.05, -128.50])
    y_sigma = np.array([0.017345454369301853, 0.0072012751512532955,
                        0.005046178740680459, 0.00043853259365807934, 0.0012702784379778687,
                        0.001618244533676047, 0.007026575545514804])
    return x_knots, y_knots, y_sigma, 7e-2


def simple_spline_fit(x, y, verbose=False):
    """Fit simplified spline-only model."""
    x_knots = np.array([2.30258509, 3.04941017, 3.79623525,
                        8.65059825, 8.95399594, 8.97733423, 9.02401079])

    def likelihood(cube, *_):
        return models.likelihood_simple_spline(
            cube, x, y, x_knots, model_spline)

    def prior(cube, *_):
        return models.prior_simple_spline(cube, x, y, x_knots)

    parameters = [r'$\sigma$']
    for i in range(len(x_knots)):
        parameters += [f'$k_{i}$']
    # out = "multinest_output/simple_"
    out = "multinest_output/tmp_"

    # Run PyMultinest
    pymultinest.run(likelihood, prior, len(parameters), n_live_points=128,
                    resume=False, verbose=True, outputfiles_basename=out,
                    sampling_efficiency="parameter")
    json.dump(parameters, open(out + "params.json", "w"))

    # Analyse output & plot results
    ana = MultiNestAnalyser(out)
    mean, sigma, bf = ana.getResults()
    print("Best fit:", bf)
    print("Mean:", mean)
    print("Sigma:", sigma)

    if not verbose:
        return x_knots, bf[1:], bf[0]
    ax = plt.subplot(111)
    ax.plot(np.exp(x), np.exp(y), label="Data", zorder=1)
    x_plot = np.logspace(np.log10(x[0]), np.log10(x[-1]), 1000)
    ax.plot(np.exp(x_plot), np.exp(model_spline(bf, x_knots)(x_plot)),
            label="Best fit", zorder=2)
    ax.scatter(np.exp(x_knots), np.exp(bf[1:len(x_knots)+1]),
               color="r", zorder=3)

    # Nice things
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(loc="best")
    plt.show()

    return x_knots, bf[1:], bf[0]


def model_interference(cube, x_data, y_data, block_positions, x_knots):
    """Fit segments for given spline params."""
    y_spline = model_spline(cube, x_knots, shift=0)(x_data)
    popt, _ = fit_segments(x_data, y_data, x_knots, cube,
                           block_positions, verbose=False)

    return y_spline + models.model_segment_line(
        popt.flatten(), x_data, block_positions, shift=0)


def interference_fit_alt():
    """Placeholder alternative to semi-combined ultranest fit."""
    return [-91.35561965873303, -97.31748429524563, -101.16588314032273, -101.97933713952926,
            -105.32623108070874, -107.0547970741264, -128.51671359120186]


def interference_fit(X, Y, x_knots, y_knots, y_sigma, sigma,
                     block_positions, pruning, verbose=False):
    """Combined fit where the line optimisations are semi-separate."""
    _X, _Y = X[::pruning], Y[::pruning]
    _block_positions = np.array([pos / pruning for pos in block_positions], dtype=int)

    def likelihood(cube, *_):
        return models.likelihood_interference(
            cube, _X, _Y, sigma, x_knots, _block_positions, model_interference)

    def prior(cube, *_):
        return models.prior_interference(cube, y_knots, y_sigma)

    # Run UltraNest
    parameters = [f'$k_{i}$' for i in range(len(x_knots))]

    # TODO: save data in log folder & retrieve!
    sampler = ultranest.ReactiveNestedSampler(
        parameters, likelihood, prior
    )
    sampler.stepsampler = ultranest.stepsampler.SliceSampler(
        nsteps=2*len(parameters),
        generate_direction=ultranest.stepsampler.generate_mixture_random_direction
    )

    results = sampler.run()
    bf = results["maximum_likelihood"]["point"]
    sampler.print_results()
    if not verbose:
        return bf

    # Plot results
    ax = plt.subplot(111)
    x, y = np.exp(X), np.exp(Y)
    ax.plot(x, y, label="Data", zorder=1)

    # Plot combined model
    x_plot = np.logspace(np.log10(x[0]), np.log10(x[-1]), 1000)
    y_model = model_spline(y_knots, x_knots, shift=0)(np.log(x_plot))
    popt, _ = fit_segments(np.log(x), np.log(y), x_knots, y_knots,
                           block_positions, verbose=False)
    pruning = int(len(x) / len(x_plot))
    _block_positions = np.array([pos / pruning for pos in block_positions], dtype=int)
    y_model += models.model_segment_line(popt.flatten(), np.log(x_plot), _block_positions, shift=0)
    ax.plot(x_plot, np.exp(y_model), label="Interference model")

    # Nice things
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(loc="upper right")

    # Plot residuals
    plt.figure()
    ax = plt.subplot(111)
    ax.set_xscale("log")
    y_model = model_spline(y_knots, x_knots, shift=0)(np.log(x))
    y_model += models.model_segment_line(popt.flatten(), np.log(x), block_positions, shift=0)
    ax.plot(x, y - np.exp(y_model))
    ax.set_title("Data - Model")
    plt.show()

    return bf


def fit_segments(x, y, x_knots, y_knots, block_positions, verbose=False):
    """Fit each LPSD segment individually with a line, return fit results."""
    def line(x, a, b):
        return a*x + b

    n_segments = len(block_positions) - 1
    popt, pcov = np.zeros((n_segments, 2)), np.zeros((n_segments, 2, 2))
    for i, start_pos in enumerate(tqdm(block_positions[:-1],
                                       desc="Prelim. segment fit",
                                       disable=not verbose)):
        end_pos = block_positions[i + 1]
        x_segment, y_segment = x[start_pos:end_pos], y[start_pos:end_pos]
        y_model = model_spline(y_knots, x_knots, shift=0)(x_segment)

        # Fit line
        a, b = np.polyfit(x_segment, y_segment - y_model, deg=1)
        popt[i, :], pcov[i, ...] = curve_fit(
            line, x_segment, y_segment - y_model, p0=[a, b], absolute_sigma=False)

        if verbose:
            plt.figure()
            plt.plot(np.exp(x_segment), y_segment - y_model)
            plt.plot(np.exp(x_segment), a*x_segment + b)
            plt.title(f"a:{a:.2e}, b:{b:.2e}")
            plt.show()

    return popt, pcov


def remove_peaks(X, Y, block_positions, x_knots, y_knots):
    """Remove peak data in residuals for further post-processing down the line."""
    residuals = Y - model_interference(y_knots, X, Y, block_positions, x_knots)
    plt.plot(X, residuals)
    plt.show()


def main():
    """Coordinate analysis."""
    # Analysis variables
    do_prelim_spline_fit = False
    do_prelim_combined_fit = False
    pruning = 1000

    # Set LPSD variables
    fmin = 10  # Hz
    fmax = 8192  # Hz
    fs = 16384  # Hz
    resolution = 1e-6
    epsilon = 0.1
    block_positions = list(block_position_gen(fmin, fmax, resolution, fs, epsilon))

    # Load data
    filename = "data/result_epsilon10_1243393026_1243509654_H1.txt"
    print("Reading data..")
    freq, psd = utils.read(filename, n_lines=utils.Jdes(fmin, fmax, resolution))
    if not psd[-1]:
        freq, psd = np.delete(freq, -1), np.delete(psd, -1)
    X, Y = np.log(freq), np.log(psd)

    # 1- Perform first simplified spline-only fit
    print("Performing preliminary spline fit..")
    if do_prelim_spline_fit:
        x_knots, y_knots, y_sigma, sigma = simple_spline_fit(X, Y, verbose=True)
    else:
        x_knots, y_knots, y_sigma, sigma = simple_spline_alt()

    # 2- Preliminary spline+lines (semi-combined) fit.
    print("Performing preliminary semi-combined fit..")
    if do_prelim_combined_fit:
        y_knots = interference_fit(X, Y, x_knots, y_knots, y_sigma, sigma,
                                   block_positions, pruning, verbose=True)
    else:
        y_knots = interference_fit_alt()

    # 3- Remove peak data and adjust block positions accordingly
    X_noise, Y_noise, block_positions_noise = remove_peaks(
        X, Y, block_positions, x_knots, y_knots)


if __name__ == '__main__':
    main()
