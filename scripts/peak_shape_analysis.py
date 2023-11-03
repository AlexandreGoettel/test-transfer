"""Find out what peak shape the signals have in the LPSD spectrum. Compare with theory."""
import csv
import h5py
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import curve_fit
from scipy.stats import norm, skewnorm
# Project imports
from peak_finder import PeakFinder
import sensutils
import hist
import utils


def fit_chunks(y_test, y_theory, chunk_size=10000, alpha_cfd=.1,
               peak_sigma=5, nbins_fit=75, verbose=False):
    """Fit skew normal through residual chunks and identify peaks by transforming normal cdf."""
    N, start = len(y_test), 0
    peak_mask = np.zeros(len(y_theory), dtype=bool)
    ends = np.arange(chunk_size, N, chunk_size)
    chis = np.zeros(len(ends))
    for i, end in enumerate(tqdm(ends, desc="Skew norm fits")):
        # Create and clean residual array
        y, y_theo = y_test[start:end], y_theory[start:end]
        residuals = y - y_theo * np.median(y) / np.median(y_theo)
        h0, bins = np.histogram(residuals, np.linspace(min(residuals), max(residuals), nbins_fit))
        mask_h0 = h0 >= alpha_cfd * max(h0)
        # Create a mask where True indicates the residual should be kept
        bin_indices = np.digitize(residuals, bins) - 1
        mask_data = np.isin(bin_indices, np.where(mask_h0)[0])
        _residuals = residuals[mask_data]
        h0, bins = np.histogram(_residuals,
                                np.linspace(min(_residuals), max(_residuals), nbins_fit))
        if len(_residuals) <= 4 or np.sum(h0 > 0) <= 4:
            peak_mask[start:end] = np.nan
            continue

        # Fit the residual distribution with a skew norm
        alpha, mu, sigma = sensutils.get_skewnorm_p0(_residuals)
        p0 = [chunk_size, mu, sigma, alpha]
        try:
            popt, _, chi = hist.fit_hist(hist.skew_gaus, _residuals, bins,
                                         p0=p0, get_chi_sqr=True)
            chis[i] = chi
        except (ValueError, RuntimeError, TypeError) as err:
            tqdm.write(f"Failed fit block {i}: {err}")
            hist.plot_hist(_residuals, bins)
            plt.show()
            peak_mask[start:end] = np.nan
            continue

        # Use percentile to identify peaks
        peak_alpha_skew = skewnorm.ppf(1 - 2*norm.cdf(-peak_sigma), popt[3],
                                       scale=popt[2], loc=popt[1])
        peak_mask[start:end] = residuals > peak_alpha_skew

        if verbose:
            ax, _ = hist.plot_func_hist(hist.skew_gaus, popt, residuals, bins)
            ax.set_title(f"{chi:.1f}")
            ax.axvline(peak_alpha_skew, color="r")
            plt.show()

        # Loop condition
        start = end

    return peak_mask, chis


def compare_masks(pm_ref, pm_inj, verbose=False):
    """Find overlapping and non-overlapping peaks, return start/end positions."""
    no_overlap = (pm_ref == 0) & (pm_inj == 1)

    if verbose:
        x = np.arange(len(pm_ref))
        plt.figure()
        overlap_peak = (pm_ref == 1) & (pm_inj == 1)
        plt.plot(x, overlap_peak)
        plt.figure()
        plt.plot(x, no_overlap)
        plt.show()

    # Convert no_overlap to peak position arrays
    rising_edges = (no_overlap == 1) & (np.roll(no_overlap, 1) == 0)
    rising_edges[0] = False
    peak_start_idx = np.where(rising_edges)[0]

    falling_edges = (no_overlap == 0) & (np.roll(no_overlap, 1) == 1)
    falling_edges[0] = False
    peak_end_idx = np.where(falling_edges)[0] - 1

    # Make sure the first and last peaks are counted ok
    if peak_end_idx[0] < peak_start_idx[0]:
        peak_end_idx = peak_end_idx[1:]
    if peak_end_idx[-1] < peak_start_idx[-1]:
        peak_start_idx = peak_start_idx[:-1]

    assert len(peak_end_idx) == len(peak_start_idx)
    return peak_start_idx, peak_end_idx


def get_injected_pos(x, idx_start, idx_end, injected_freqs):
    """If an injected frequency is found within the bound, return it."""
    # f_start, f_end = x[idx_start], x[idx_end]
    candidates = injected_freqs[(injected_freqs >= x[idx_start]) & (injected_freqs < x[idx_end])]
    if len(candidates) == 1:
        return candidates[0]
    return 0


def align_peaks(freq, psd, peak_idx_start, peak_idx_end, injected_freqs,
                _SUFFIX=30, _BUFFER=30, _OFFSET=5, _PREFIX=15, _CFD_FRAC=.95):
    """Align peaks using CFD."""
    assert peak_idx_end[0] > peak_idx_start[0]
    max_peak_width = max(peak_idx_end - peak_idx_start)
    peak_data = np.zeros((len(peak_idx_end), max_peak_width+_SUFFIX+_PREFIX))

    # Use CFD to align peaks
    peak_frequencies = np.zeros((2, len(peak_idx_end)))
    for i, (idx_start, idx_end) in enumerate(zip(peak_idx_start, peak_idx_end)):
        # 1. Add buffer to cover enough points
        dx = max_peak_width - (idx_end - idx_start)
        imin, imax = idx_start - dx//2 - _BUFFER - _SUFFIX, idx_end + dx//2 + _BUFFER + _SUFFIX
        # 2. Get baseline
        baseline = np.median(psd[imin-_BUFFER:imin+_BUFFER])
        # 3. Get CFD x-coordinate
        cfd_target = baseline + _CFD_FRAC*np.abs(max(psd[imin:imax]) - baseline)
        idx_cfd = imin + np.where(psd[imin:imax] > cfd_target)[0][0]

        y2, x2 = psd[idx_cfd], freq[idx_cfd]
        y1, x1 = psd[idx_cfd - 1], freq[idx_cfd - 1]
        a = (y2 - y1) / (x2 - x1)
        b = (x2*y1 - x1*y2) / (x2 - x1)
        x_cfd = (cfd_target - b) / a
        # 4. Align x_CFD over saved section
        f = interp1d(freq[imin:imax], psd[imin:imax] - baseline)

        delta = (x_cfd - freq[idx_start]) / _OFFSET
        x = np.arange(freq[idx_start]-delta*_PREFIX,
                      freq[idx_start]+delta*(max_peak_width+_SUFFIX),
                      delta)
        peak_data[i, :] = f(x[:max_peak_width+_SUFFIX+_PREFIX])

        # Find peak in injection file
        peak_frequencies[0, i] = get_injected_pos(freq, idx_start, idx_end, injected_freqs)
        peak_frequencies[1, i] = x[0]
    return peak_data, peak_frequencies


def main_v2(injection_path="injections/sine/injected_data_24.lpsd",
            reference_path="injections/injected_data_noise.lpsd",
            verbose=False, from_scratch=True):
    """Extract the peak shape from an injection file's preprocessing results."""
    # Define analysis variables
    kwargs_fit = {
        "chunk_size": 10000,
        "peak_sigma": 5,
        "alpha_cfd": .1,
        "nbins_fit": 100,
        # "max_chi": 10,
        "verbose": False
    }
    kwargs_align = {
        "_CFD_FRAC": .95,
        "_BUFFER": 30,
        "_OFFSET": 5,
        "_SUFFIX": 30,
        "_PREFIX": 15
    }

    # Get data
    injected_freqs = []
    with open("data/injection_files/injections_full_1.0e-17.dat", "r") as _f:
        data = csv.reader(_f, delimiter="\t")
        for row in data:
            if row:
                injected_freqs.append(float(row[0]))
    injected_freqs = np.array(injected_freqs)

    dfile_ref = h5py.File(sensutils.get_corrected_path(reference_path))
    x_inj, y_inj = utils.read(injection_path)
    x_ref, y_ref = dfile_ref["frequency"][()], dfile_ref["logPSD"][()]
    assert len(x_ref) == len(x_inj)

    # Correct for injection bug
    y_inj = 30.5 + np.log(y_inj)
    y_ref = 30.5 + y_ref
    # Add theoretical PSD shape
    # Careful to update this with new data # TODO
    xKnots = np.array([2.30258509, 3.04941017, 3.79623525,
                    8.65059825, 8.95399594, 8.97733423, 9.02401079])
    yKnots = np.array([-91.80878694876485, -99.97801940114547, -103.57729069085298,
                    -102.17121965438, -104.34025547329, -105.9256036217, -130.06995841416])
    y_theory = CubicSpline(xKnots, yKnots)(np.log(x_inj))

    if verbose:
        ax = plt.subplot(111)
        ax.plot(x_ref, y_ref, label="Reference", alpha=.33)
        ax.plot(x_inj, y_inj, label="Injection", alpha=.33)
        ax.plot(x_inj, y_theory, label="PSD")
        ax.set_xscale("log")
        ax.legend(loc="best")
        plt.show()

    # Goal: Find peaks in inj and ref by comparing residuals using theory shape
    if from_scratch:
        peak_mask_inj, chis_inj = fit_chunks(y_inj, y_theory, **kwargs_fit)
        peak_mask_ref, chis_ref = fit_chunks(y_ref, y_theory, **kwargs_fit)
        plt.figure()
        x = np.arange(len(chis_inj))
        plt.scatter(x, chis_inj, label="Injection")
        plt.scatter(x, chis_ref, label="Reference")
        plt.legend(loc="best")
        plt.title("Skew norm chi square / dof.")
        np.savez("peak_masks.npz", ref=peak_mask_ref, inj=peak_mask_inj)
    else:
        pm_data = np.load("peak_masks.npz")
        peak_mask_ref, peak_mask_inj = pm_data["ref"], pm_data["inj"]

    # Get aligned peaks
    peak_start, peak_end = compare_masks(peak_mask_ref, peak_mask_inj)
    peak_data, peak_freqs = align_peaks(x_inj, y_inj, peak_start, peak_end,
                                        injected_freqs, **kwargs_align)

    # Quick check if some injections were missed
    missed_freqs = []
    for freq in injected_freqs:
        if freq not in peak_freqs[0, :]:
            missed_freqs.append(freq)
    score = 100*len(missed_freqs)/peak_freqs.shape[1]
    print("Missed frequencies (Hz):", np.sort(missed_freqs))
    print(f"Percentage of frequencies that were not recovered: {score:.2f}%")

    # TMP
    plt.figure()
    m = peak_freqs[0, :] != 0
    plt.plot((peak_freqs[0, m] - peak_freqs[1, m]) / peak_freqs[0, m])
    plt.xlabel("(f_inj - x_0) / f_inj")

    # Filter # TO TUNE # TODO
    peak_data = peak_data[peak_data[:, 20] > 10, :]

    # Plot & fit
    mu = np.median(peak_data, axis=0)
    sigma = np.std(peak_data, ddof=1, axis=0)
    plt.figure()
    ax = plt.subplot(111)
    x = np.arange(peak_data.shape[1])

    ax.plot(x, peak_data[0, :], color="C0", alpha=.33, zorder=0, label="Raw Data")
    for i in range(1, peak_data.shape[0]):
        ax.plot(x, peak_data[i, :], color="C0", alpha=.33, zorder=0)

    ax.fill_between(x, mu+sigma, mu-sigma, color="C1", alpha=.33, zorder=1)
    ax.plot(x, mu, color="C1", linewidth=2., zorder=1, label="Median (log)")

    def fitFunc(x, A, mu, sigma, b):
        # return A/x/np.sqrt(2*np.pi)/sigma*np.exp(-0.5*((np.log(x) - mu) / sigma)**2) + b
        return A/np.sqrt(2*np.pi*sigma)*np.exp(-0.5*((x - mu) / sigma)**2) + b
    popt, _= curve_fit(fitFunc, x[1:], mu[1:], p0=[35, 20, 0.64, -105],
                       sigma=sigma[1:], absolute_sigma=True)

    ax.plot(x, fitFunc(x, *popt), color="C2", label="Gaus fit", zorder=3)

    # Nice things
    ax.set_xlabel("(a.u.)")
    ax.set_ylabel("ln(PSD)")
    ax.legend(loc="upper right")
    plt.show()

    # Save normalised peak data
    y = mu[12:35] / np.sum(mu[12:35])
    np.save("peak_shape.npy", y)  # Use as template fit


if __name__ == '__main__':
    main_v2(from_scratch=False)
    # main_v2(injection_path="data/result_epsilon10_1261859291_1262112768_L1.lpsd",
    #         reference_path="data/result_epsilon10_1252506806_1252651567_L1.hdf5")
