"""Fit background (L)PSD with spline + segment lines."""
# Standard imports
import os
import json
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
    do_prelim_interference_fit = True

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
    pruning = 1000
    pf.combined_fit(peak_mask, block_positions, x_knots, y_knots, y_sigma,
                    popt_segments, pcov_segments, pruning=pruning)

    # 6- Plot results from full combined fit
    X, Y = np.log(pf.freq[~peak_mask][::pruning]), np.log(pf.psd[~peak_mask][::pruning])
    block_positions = pf.adjust_block_positions(block_positions, ~peak_mask)
    block_positions = np.array([pos / pruning for pos in block_positions], dtype=int)
    test = np.concatenate([y_knots, popt_segments.flatten()])
    print(models.likelihood_combined(test, X, Y, x_knots, block_positions))

    with open("out_test/info/results.json") as _f:
        results = json.load(_f)
    bf = np.array(results["maximum_likelihood"]["point"])

    model_before = models.model_combined(test, X, block_positions, x_knots)
    model_after = models.model_combined(bf, X, block_positions, x_knots)

    plt.figure()
    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(np.exp(X), np.exp(Y))
    ax.plot(np.exp(X), np.exp(model_before), label="Interference fit")
    ax.plot(np.exp(X), np.exp(model_after), label="Combined fit")
    ax.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    main()
