"""Analyse peaks for compatibility with dark matter."""
import os
import numpy as np
from matplotlib import pyplot as plt
import utils
import hist
from peak_finder import PeakFinder


def main():
    # Set analysis variables
    epsilon = 10
    t0, t1 = 1243393026, 1243509654
    ifo = "H1"
    fmin = 10  # Hz
    fmax = 8192  # Hz
    resolution = 1e-6

    # Set derivative variables
    g = np.log(fmax) - np.log(fmin)
    J = utils.Jdes(fmin, fmax, resolution)

    # Get data
    suffix = f"epsilon{epsilon}_{t0}_{t1}_{ifo}"
    _, psd = utils.read(os.path.join("data", f"result_{suffix}.txt"),
                        raw_freq=False)
    freq = fmin * np.exp(np.arange(J) * g / float(J - 1)) *\
        (np.exp(g / float(J - 1)) - 1)
    peak_info = np.load(os.path.join("data", f"peak_info_{suffix}.npy"))
    # data = (freq[1:] - freq[:-1]) / freq[:-1]
    # bins = np.logspace(np.log10(min(data)), np.log10(max(data)), 1000)
    # ax = hist.plot_hist(data, bins, logy=True, logx=True)
    # plt.show()
    # return

    # Clean peak info
    peak_info = peak_info[~np.isinf(peak_info[:, 2]), ...]
    snr_cut = 5

    deltas = np.zeros(len(peak_info), dtype=np.float64)
    for i, (start, width, _) in enumerate(peak_info):
        if start + width >= len(freq):
            continue
        start, width = int(start), int(width)
        deltas[i] = (freq[start+width] - freq[start]) /\
            (freq[start] + freq[start + width]) * 2.

    print(peak_info.shape)
    mask = (peak_info[:, 2] >= snr_cut) & (deltas != 0.)
    print(peak_info[mask, ...].shape)
    # (deltas <= max_delta) & (deltas != 0.)
    # (peak_info[:, 1] <= max_peak_width) &\

    # Cut plots
    high_snr_peaks = peak_info[peak_info[:, 2] > snr_cut, :]
    _deltas = deltas[deltas != 0.]
    bins = np.logspace(np.log10(min(_deltas)), np.log10(max(_deltas)), 100)
    ax = hist.plot_hist(_deltas, bins, logx=True, logy=True, label="Raw data")
    ax = hist.plot_hist(deltas[mask], bins, ax=ax, label="High SNR")
    ax.legend(loc="best")
    ax.set_xlabel("Delta")
    ax = hist.plot_hist(peak_info[:, 1], np.arange(max(peak_info[:, 1])),
                        logy=True, label="Raw data")
    ax = hist.plot_hist(high_snr_peaks[:, 1], np.arange(max(high_snr_peaks[:, 1])),
                        ax=ax, label="High SNR peaks")
    ax.legend(loc="upper right")
    ax.set_xlabel("Peak width in bins")
    ax = hist.plot_hist(peak_info[:, 2], 300, logy=True)
    ax.set_xlabel("Peak SNR")
    plt.show()

    buffer = 100
    for delta, (start, width, height) in zip(deltas[mask], peak_info[mask, ...]):
        if delta > 6e-6:
            continue
        start, end = int(start - buffer), int(start + width + buffer)
        plt.figure()
        ax = plt.subplot(111)
        ax.set_yscale("log")
        ax.plot(freq[start:end], psd[start:end])
        ax.set_title(f"SNR: {height:.1f}, width: {int(width)} bins, " +
                     f"delta: {delta:.1e}")
        y0, y1 = ax.get_ylim()
        ax.fill_between([freq[int(start+buffer)], freq[int(start+buffer+width)]],
                        [y1, y1], [y0, y0], color="C3", alpha=.33)
        ax.set_ylim(y0, y1)
        plt.show()


if __name__ == '__main__':
    main()
