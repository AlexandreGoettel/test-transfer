"""Fit background (L)PSD with spline + segment lines."""
# Standard imports
import os
import numpy as np
from matplotlib import pyplot as plt
# Project imports
from peak_finder import PeakFinder
import models


def simple_spline_alt():
    """Temporary alternative to launching multinest."""
    x_knots = np.array([2.30258509, 3.04941017, 3.79623525,
                        8.65059825, 8.95399594, 8.97733423, 9.02401079])
    y_knots = np.array([-91.365, -97.333, -101.16,
                        -101.98, -105.33, -107.05, -128.50])
    y_sigma = np.array([0.017345454369301853, 0.0072012751512532955,
                        0.005046178740680459, 0.00043853259365807934, 0.0012702784379778687,
                        0.001618244533676047, 0.007026575545514804])
    return x_knots, y_knots, y_sigma, 7e-2


def interference_fit_alt():
    """Temporary alternative to semi-combined ultranest fit."""
    _data = np.load("out/prelim_fit.npz")
    return _data["y_knots"], _data["popt_segments"], _data["pcov_segments"]


def main():
    """Run full-chain analysis on LPSD data segment."""
    # Set analysis constraints
    do_prelim_spline_fit = False
    do_prelim_interference_fit = False

    # Temporary analysis-wide constants
    LOWER_LIM, UPPER_LIM = -5, 3  # Prelim cut
    _SEGMENT_SIZE = int(1e4)  # Size of chunks for skew norm fits
    _CFD_ALPHA = .05  # CFD threshold for peak zoom in fits
    _CHI_LOW, _CHI_HIGH = .1, 2  # Quality cuts on fit results
    _CUT_ALPHA = .99  # How much of the whitened norm data to keep

    # Set LPSD variables
    kwargs = {"fmin": 10,  # Hz
              "fmax": 8192,  # Hz
              "fs": 16384,  # Hz
              "resolution": 1e-6,
              "epsilon": 0.1,
              "name": "data/result_epsilon10_1243393026_1243509654_H1.txt"
              }
    pf = PeakFinder(**kwargs)

    # 1- Perform first simplified spline-only fit
    print("Performing preliminary spline fit..")
    x_knots, y_knots, y_sigma, sigma = pf.simple_spline_fit()\
        if do_prelim_spline_fit else simple_spline_alt()

    # 2- Preliminary spline+lines (semi-combined) fit.
    print("Performing preliminary semi-combined fit..")
    block_positions = list(pf.block_position_gen())
    if do_prelim_interference_fit:
        y_knots, popt_segments, pcov_segments = pf.interference_fit(
            x_knots, y_knots, y_sigma, sigma,
            block_positions, 1000, verbose=True)
        np.savez("out/prelim_fit.npz",
            y_knots=y_knots,
            popt_segments=popt_segments,
            pcov_segments=pcov_segments)
    else:
        y_knots, popt_segments, pcov_segments = interference_fit_alt()

    # 3- Fit skew normals across frequency blocks, apply preliminary peak mask
    print("Fit chunks with skew normals..")
    Y, X = np.log(pf.psd), np.log(pf.freq)
    residuals = np.log(pf.psd) - models.model_interference(
        y_knots, X, Y, block_positions, x_knots, pf.fit_segments
    )
    del X, Y
    peak_mask = (residuals < LOWER_LIM) ^ (residuals > UPPER_LIM)
    popt, _, chis = pf.fit_frequency_blocks(
        residuals[~peak_mask],
        _CFD_ALPHA, _SEGMENT_SIZE, _CHI_LOW, _CHI_HIGH)

    # 4- Whiten residuals to find peaks
    print("Whiten data and identify peaks..")
    peak_mask, peak_info = pf.get_peak_mask_from_residuals(
        residuals, peak_mask, popt, chis,
        _SEGMENT_SIZE, _CHI_LOW, _CHI_HIGH,
        _CUT_ALPHA, verbose=True)
    prefix = "_".join(os.path.split(kwargs["name"])[-1].split("_")[1:]).split(".", maxsplit=1)[0]
    np.save(os.path.join("data", f"peak_info_{prefix}.npy"), peak_info)

    # 5- Combined fit on cleaned data
    pruning = 10000
    # pf.combined_fit(peak_mask, block_positions, x_knots, y_knots, y_sigma,
    #                 popt_segments, pcov_segments, pruning=pruning)
    initial_guess = np.concatenate([y_knots, popt_segments.flatten()])
    bf = pf.combined_fit_minimize(peak_mask, block_positions, pruning,
                                  x_knots, initial_guess)

    ####################
    # Second iteration # starting data has peak_mask applied
    ####################
    # TODO: incorporate loop
    # 2.3 Fit skew normals along residuals
    Y, X = np.log(pf.psd), np.log(pf.freq)
    # _block_positions = pf.adjust_block_positions(block_positions, ~peak_mask)
    residuals = np.log(pf.psd) - models.model_combined(
        bf, X, block_positions, x_knots)
    popt, _, chis = pf.fit_frequency_blocks(
        residuals[~peak_mask],
        _CFD_ALPHA, _SEGMENT_SIZE, _CHI_LOW, _CHI_HIGH)

    # 2.4 Whiten data and find peaks
    peak_mask, peak_info = pf.get_peak_mask_from_residuals(
        residuals, peak_mask, popt, chis,
        _SEGMENT_SIZE, _CHI_LOW, _CHI_HIGH,
        _CUT_ALPHA, verbose=True)
    prefix = "_".join(os.path.split(kwargs["name"])[-1].split("_")[1:]).split(".", maxsplit=1)[0]
    np.save(os.path.join("data", f"peak_info_{prefix}.npy"), peak_info)

    # 2.5 Combined fit
    initial_guess = bf
    pruning = 10000
    bf = pf.combined_fit_minimize(peak_mask, block_positions, pruning,
                                  x_knots, initial_guess)
    print(bf)

    # PLOT RESULTS - TMP
    _X, _Y = np.log(pf.freq[~peak_mask][::pruning]), np.log(pf.psd[~peak_mask][::pruning])
    _block_positions = pf.adjust_block_positions(block_positions, ~peak_mask)
    _block_positions = np.array([pos / pruning for pos in _block_positions], dtype=int)

    print(models.likelihood_combined(initial_guess, _X, _Y, x_knots, _block_positions))
    print(models.likelihood_combined(bf, _X, _Y, x_knots, _block_positions))
    model_before = models.model_combined(
        initial_guess, X, block_positions, x_knots)
    model_after = models.model_combined_vec(
        bf[None, :], X, block_positions, x_knots)[0, :]

    # Plot effect of second iteration on bkg fit
    plt.figure()
    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(np.exp(X[~peak_mask]), np.exp(Y[~peak_mask]))
    ax.plot(np.exp(X), np.exp(model_before), label="First iteration")
    ax.plot(np.exp(X), np.exp(model_after), label="Second iteration")
    ax.legend(loc="upper right")

    # Plot cleaned data with residual spline component
    plt.figure()
    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(np.exp(X), np.exp(Y - models.model_segment_line(
        bf, X, block_positions, shift=len(x_knots))))
    plt.show()


if __name__ == '__main__':
    main()
