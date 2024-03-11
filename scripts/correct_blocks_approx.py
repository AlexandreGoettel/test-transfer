"""Use a hybrid fit to correct for block structure from the const-N approximation."""
import os
import argparse
import glob
import h5py
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
from scipy.stats import norm
# Project imports
from LPSDIO import LPSDOutput
import models
import hist
import stats



def parse_args():
    """Define cmdl. arg. parser and return vars."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to lpsd output file or folder of files.")
    parser.add_argument("--data-prefix", type=str, default="",
                        help="If data-path is a dir, only consider files starting with prefix.")
    parser.add_argument("--buffer", type=int, default=50,
                        help="Interval to median over for knot creation.")
    parser.add_argument("--segment-size", type=int, default=int(1e4),
                        help="Size of skew-norm fit interval")
    parser.add_argument("--alpha-CFD", type=float, default=.05,
                        help="CFD ratio to remove peak influence in skew norm fits.")
    parser.add_argument("--alpha-cut", type=float, default=.99,
                        help="Percentage of data to keep (assuming norm) during peak removal.")
    parser.add_argument("--chi-lo", type=float, default=.01,
                        help="Minimal accepted chi^2/dof value in skew norm fits.")
    parser.add_argument("--chi-hi", type=float, default=2,
                        help="Maximal accepted chi^2/dof value in skew norm fits.")
    parser.add_argument("--bin-factor", type=float, default=.01,
                        help="Inverse bin width in bin units for skew norm fits.")
    parser.add_argument("--pruning", type=int, default=1000,
                        help="Pruning to apply to data during simul. fit / plotting if verbose")
    parser.add_argument("--verbose", action="store_true")

    return vars(parser.parse_args())


def get_block_corrected_path(data_path):
    """Generate path to store block-corrected results."""
    return os.path.join(
        os.path.split(data_path)[0],
        "block_corrected_"
        + ".".join(os.path.split(data_path)[1].split(".")[:-1])
        + ".hdf5"
    )


def apply_cfd(data, bins, _cfd):
    """Apply rising and falling edge constant fraction discriminator."""
    h0, bins = np.histogram(data, bins)
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


def seq_mean(x, y):
    """Calculate the mean over a sequence (e.g. hist.)"""
    return np.sum(x * y) / np.sum(y)


def seq_var(x, y):
    """Calculate the variance over a sequence (e.g. hist.)"""
    mu = seq_mean(x, y)
    return np.sum(y*(x - mu)**2) / np.sum(y)


def get_peak_info(data, mask):
    """Return the start position, width, and max height of peaks where values exceed a cut."""
    # Find the starting and ending points of the peaks
    extended_mask = np.concatenate(([False], mask, [False]))
    change_points = np.diff(extended_mask.astype(int))

    # find the starting and ending indices of each peak
    peak_starts = np.where(change_points == 1)[0]
    peak_ends = np.where(change_points == -1)[0]
    assert peak_starts.size == peak_ends.size

    # Calculate the widths and heights of the peaks
    widths = peak_ends - peak_starts
    max_heights = [data[start:end].max() for start, end in zip(peak_starts, peak_ends)]
    return np.stack((peak_starts, widths, max_heights), axis=-1)


class BlockCorrector:
    """Correct block structure from constant-N approximation."""

    def __init__(self, data, **kwargs):  # LPSDData
        assert all([flag in kwargs for flag in
                    ["segment_size",
                     "chi_lo",
                     "chi_hi",
                     "alpha_CFD",
                     "bin_factor",
                     "alpha_cut"]])
        self.__dict__.update(kwargs)

        self.data = data
        self.block_positions = list(self.block_position_gen())

    def block_position_gen(self):
        """Generator for block-approximation edge positions."""
        j0, j, Jdes = 0, self.data.vars.j(0), self.data.vars.Jdes
        yield 0  # Always start at 0
        yield int(j)
        while True:
            j0 = j
            j = self.data.vars.j(j0)
            if j >= Jdes - 1:
                yield Jdes - 2
                return
            yield int(j)

    def adjust_block_positions(self, mask, pruning=1000):
        """Given block positions and a mask of to-be-removed indices, adjust."""
        cumulative_mask = np.cumsum(mask)
        cumulative_mask = np.append(cumulative_mask, cumulative_mask[-1])
        return np.array([pos / pruning for pos in cumulative_mask[self.block_positions] - 1],
                        dtype=int)

    def line_only_fit(self, x_knots, y_knots, pruning=1000, verbose=False):
        """Fit line-only slopes on existing spline fit data."""
        X = np.log(self.data.freq[::pruning])
        Y = self.data.logPSD[::pruning]
        block_positions = self.adjust_block_positions(
            np.ones(len(self.data), dtype=int), pruning=pruning)
        y_spline = models.model_spline(y_knots, x_knots, shift=0, extrapolate=True)(X)

        n_segments = len(block_positions) - 1
        popt, pcov = np.zeros(n_segments), np.zeros((n_segments, 1, 1))
        for i, start_pos in enumerate(tqdm(block_positions[:-1],
                                        desc="Segment only fits")):
            end_pos = block_positions[i + 1]
            x_segment, y_segment = X[start_pos:end_pos], Y[start_pos:end_pos]
            y_model = models.model_spline(y_knots, x_knots, shift=0, extrapolate=True)(x_segment)
            # Fit line
            a, _ = np.polyfit(x_segment, y_segment - y_model, deg=1)
            def line(x, slope):
                offset = -a*x_segment[0]
                return slope*x + offset

            try:
                popt[i], pcov[i, ...] = curve_fit(
                    line, x_segment, y_segment - y_model, p0=[a], absolute_sigma=False)
            except ValueError:
                tqdm.write(f"Skipping fit #{i+1}..")
                continue

            if verbose:
                y_spline[start_pos:end_pos] +=\
                    popt[i]*x_segment - popt[i]*x_segment[0]

        if verbose:
            plt.figure()
            plt.plot(X, Y, label="Data")
            plt.plot(X, y_spline, label="Fit")
            plt.plot(X, models.model_spline(y_knots, x_knots, shift=0, extrapolate=True)(X),
                     label="Spline only")
            plt.legend(loc="best")
            plt.show()
        return popt, pcov

    def combine_spline_slope_smart(self, x_knots, y_knots, peak_mask, initial_guess,
                                   pruning=1000, verbose=False):
        """Combine spline and slope fit based on initial_guess."""
        X = np.log(self.data.freq[~peak_mask][::pruning])
        Y = self.data.logPSD[~peak_mask][::pruning]
        block_positions = self.adjust_block_positions(~peak_mask, pruning=pruning)

        # Optimise
        def lkl(cube, *_):
            y_spline = models.model_spline(cube, x_knots, shift=3)(X)
            y_lines = models.model_segment_slope(cube, X, block_positions,
                                                 shift=3+len(x_knots))
            lkl = stats.logpdf_norm(Y - y_spline - y_lines,
                                    scale=1e-4 + np.abs(cube[0]*X*X + cube[1]*X + cube[2]))
            return lkl.sum()
        p0 = np.concatenate([[0, 0, 1], y_knots, initial_guess])
        tqdm.write("Starting minimisation for combined spline+slopes fit..")
        popt_combined = minimize(lambda x: -lkl(x), p0, method="Powell")
        tqdm.write("Done!")

        try:  # Debugging
            assert popt_combined.success
        except AssertionError as e:
            print(popt_combined)
            raise e

        bestfit = popt_combined.x
        if verbose:
            plt.figure()
            plt.plot(X, Y, label="Data")
            y_spline = models.model_spline(bestfit, x_knots, shift=3)(X)
            y_lines = models.model_segment_slope(bestfit, X, block_positions,
                                                 shift=3+len(x_knots))
            plt.plot(X, y_spline, label="Spline only")
            plt.plot(X, y_spline + y_lines, label="Best-fit")
            plt.legend(loc="best")
            deviation = np.power(Y - y_spline - y_lines, 2).sum()
            plt.title(f"{deviation:.2f}")
            plt.grid(linestyle="--", color="grey", alpha=.5)

            plt.figure()
            ax = plt.subplot(111)
            ax.plot(np.exp(X), np.exp(Y - y_lines), label="Corrected PSD")
            ax.plot(np.exp(X), np.exp(y_spline), label="Spline fit",
                    linestyle="-.", linewidth=1.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(axis="y", linestyle="--", color="grey", alpha=.33, linewidth=1.5,
                    which="major")
            ax.grid(axis="x", linestyle="--", color="grey", alpha=.33, linewidth=1.5,
                    which="both")
            ax.minorticks_on()
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power Spectral Density")
            ax.legend(loc="best")
            plt.tight_layout()

            plt.figure()
            plt.plot(X, bestfit[0]*X*X + bestfit[1]*X + bestfit[2])
            plt.title("Gaussian sigma vs frequency")

            plt.show()

        # Perform chi^2 fit for an uncertainty estimate
        def fitfunc(x, *args):
            y_spline = models.model_spline(args, x_knots, shift=0)(x)
            y_lines = models.model_segment_slope(args, x, block_positions,
                                                 shift=0+len(x_knots))
            return y_spline + y_lines

        _, pcov = curve_fit(fitfunc, X, Y, p0=bestfit[3:], absolute_sigma=False,
                            sigma=1e-4 + np.abs(bestfit[0]*X*X + bestfit[1]*X + bestfit[2]))

        X, Y = np.log(self.data.freq), self.data.logPSD
        y_model = models.model_spline(bestfit, x_knots, shift=3)(X)
        y_model += models.model_segment_slope(bestfit, X, block_positions, shift=3+len(x_knots))
        return bestfit, np.sqrt(np.diag(pcov)), y_model

    def do_skew_norm_fit(self, data, bin_factor=.02, cfd_factor=.05, verbose=False):
        """'Zoom' and fit around peak using a small CFD to avoid being disturbed by outliers."""
        bins = np.linspace(min(data), max(data), int(len(data)*bin_factor))
        bins = apply_cfd(data, bins, cfd_factor)

        # Estimate starting parameters
        h0, bins = np.histogram(data, bins)
        bin_centers = (bins[1:] + bins[:-1]) / 2.
        peak_index = np.argmax(h0)
        p0 = [1, bins[peak_index], np.sqrt(seq_var(bin_centers, h0)), 0]
        p0[0] = h0[peak_index] / (stats.pdf_skewnorm(bins[peak_index], *p0) *\
            (bins[1] - bins[0]))

        # Perform fit
        try:
            popt, pcov, chi = hist.fit_hist(stats.pdf_skewnorm, data, bins,
                                            p0=p0, get_chi_sqr=True)
        except (ValueError, RuntimeError):
            return np.zeros_like(p0), np.zeros((len(p0), len(p0))), np.inf

        if verbose:
            ax, axRes = hist.plot_func_hist(stats.pdf_skewnorm, popt, data, bins,
                                            label="Skew norm fit")
            ax.legend(loc="best")
            axRes.set_xlabel("log(PSD) - model")
            ax.set_title(r"$\chi^2$/ndof" + f": {chi:.2f}")
            plt.show()

        return popt, pcov, chi

    def get_chunk_lims(self, _end, segment_size):
        """Loop over data chunks (in f space) of length SEGMENT_SIZE."""
        ends = np.arange(segment_size, _end, segment_size)
        for _start, _end in zip(ends - segment_size, ends):
            if _end == ends[-1]:
                _start = _end - segment_size
            yield (_start, _end)

    def fit_frequency_blocks(self, residuals, verbose=False):
        """Separate residuals in blocks and fit each block's projection with a skew norm."""
        chis = np.zeros(int(len(residuals) / self.segment_size)+1)
        popt, pcov = np.zeros((chis.shape[0], 4)), np.zeros((chis.shape[0], 4, 4))
        for i, (start, end) in enumerate(tqdm(
            self.get_chunk_lims(len(residuals), self.segment_size),
                total=len(chis), desc="Skew norm fits..")):
            # Perform fit - the CFD lets us ignore peak influence
            _residuals = residuals[start:end]
            popt[i, :], pcov[i, ...], chis[i] = self.do_skew_norm_fit(
                _residuals, bin_factor=self.bin_factor,
                cfd_factor=self.alpha_CFD, verbose=verbose)
            if chis[i] > self.chi_hi:
                popt[i, :], pcov[i, ...], chis[i] = self.do_skew_norm_fit(
                    _residuals, bin_factor=3*self.bin_factor,
                    cfd_factor=2*self.alpha_CFD, verbose=verbose)
            if chis[i] < self.chi_lo:
                popt[i, :], pcov[i, ...], chis[i] = self.do_skew_norm_fit(
                    _residuals, bin_factor=self.bin_factor,
                    cfd_factor=2*self.alpha_CFD, verbose=verbose)

        if verbose:
            plt.figure()
            plt.scatter(np.arange(len(chis)), chis)
            plt.title("Chi^2/dof")
            plt.show()
        return popt, pcov, chis

    def get_peak_mask_from_residuals(self, residuals, peak_mask, popt, chis,
                                     pruning=100, verbose=False):
        """Whiten data using skewnorm fit results and apply cut to find peaks."""
        # Filter good fits with a cut on chi^2/ndof
        goodFit = (chis > self.chi_lo) & (chis < self.chi_hi)
        # If necessary, interpolate between valid fit results
        firstValid, lastValid = np.where(goodFit)[0][0], np.where(goodFit)[0][-1]
        interp_funcs = []
        for i in range(popt.shape[1] - 1):
            interp_funcs += [interp1d(
                np.arange(chis.shape[0])[goodFit],
                popt[:, i+1][goodFit]
                )]

        # Whitening step - must replicate the looping from the fit
        plot_x, plot_y = [], []
        whitened_data = np.zeros_like(residuals)
        chunk_limits = list(self.get_chunk_lims(len(residuals), self.segment_size))

        # Add variables to track chunks in original residuals data
        peak_cumsum = np.cumsum(~peak_mask) - 1
        N, _count, _previous = len(residuals), 0, 0
        for i, (start, end) in enumerate(tqdm(chunk_limits[firstValid:lastValid+1],
                                         desc="Whitening..")):
            # Loop condition
            n_loop = end - start
            while _count < N and peak_cumsum[_count] < start + n_loop:
                _count += 1
            _start = _previous
            _end, _previous = _count, _count
            # Whiten full data using _start and _end mapping
            mu, sigma, alpha = list(map(lambda f: f(i), interp_funcs))
            whitened_data[_start:_end] = norm.ppf(
                stats.cdf_skewnorm(residuals[_start:_end], loc=mu, scale=sigma, alpha=alpha)
            )
            # Prepare plot
            plot_x.extend(self.data.freq[_start:_end])
            plot_y.extend(whitened_data[_start:_end])

        # Get positive peak information for later DM searches.
        cut_value = -norm.ppf((1. - self.alpha_cut) / 2.)
        peak_info = get_peak_info(whitened_data, whitened_data >= cut_value)

        # Create second mask to mask all deviations and invalid values
        _full_mask = (peak_mask == 1) ^ (whitened_data < -cut_value)
        _full_mask[~peak_mask] = whitened_data[~peak_mask] >= cut_value

        # Remove start and end bad fit chunk
        _full_mask[:chunk_limits[firstValid][0]] = True
        _full_mask[chunk_limits[lastValid][1]:] = True

        if verbose:
            plt.figure()
            ax = plt.subplot(111)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.plot(self.data.freq[::pruning],
                    np.exp(self.data.logPSD[::pruning]),
                    label="Original")
            ax.plot(self.data.freq[~_full_mask][::pruning],
                    np.exp(self.data.logPSD[~_full_mask][::pruning]),
                    label="Non-peak")
            ax.legend(loc="best")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD")
            plt.show()
        return _full_mask, peak_info


def correct_blocks(data_path, verbose=False, recursion_threshold=1,
                   buffer=50, pruning=1000, **kwargs):
    """
    Corrects data for block-approximation drift and save the output to HDF.

    Parameters
    ----------
    data_path : str
        The file path to the data that needs to be corrected.
    recursion_threshold : int, optional
        The threshold for recursion depth in the correction algorithm. Defaults to 1.
    buffer : int, optional
        The buffer size to use while reading blocks of data for correction. Defaults to 50.
    pruning : int, optional
        pruning to use in simultaneous slope+spline fit and plots, for speed
    verbose : bool, optional
    **kwargs : dict
        Additional kwargs for BlockCorrector
    """
    output_path = get_block_corrected_path(data_path)
    if os.path.exists(output_path) and\
            input(f"File '{output_path}' already exists, continue? (y/n) ").lower() != "y":
        return
    data = LPSDOutput(data_path)
    corrector = BlockCorrector(data, **kwargs)

    # Get knots from block positions
    x_knots = np.array([np.log(data.freq[pos])
                        for pos in corrector.block_positions[:-1]] + [np.log(data.freq[-1])])
    y_knots = np.array([np.median(data.logPSD[pos:pos+buffer])
                        for pos in corrector.block_positions[:-1]]
                       + [data.logPSD[-1]])

    # Fit slopes on top of static spline
    popt, _ = corrector.line_only_fit(x_knots, y_knots, pruning=pruning, verbose=verbose)

    # Combined fit using previous result as initial guess
    bestfit, _, Y_model = corrector.combine_spline_slope_smart(
        x_knots, y_knots, np.zeros(len(data), dtype=bool), popt, pruning=pruning, verbose=verbose)

    # Fit residuals with skew normal in chunks
    residuals = corrector.data.logPSD - Y_model
    del Y_model
    popt_chunks, _, chis_chunks = corrector.fit_frequency_blocks(residuals)

    # Whiten fitted residuals to find peaks
    peak_mask, _ = corrector.get_peak_mask_from_residuals(
        residuals, np.zeros(len(data), dtype=bool), popt_chunks, chis_chunks,
        verbose=verbose, pruning=pruning)

    # Now fit on peak-less data (we are fitting the background after all)
    new_bestfit, sigma, Y_model = corrector.combine_spline_slope_smart(
        x_knots, y_knots, peak_mask, bestfit[3+len(x_knots):], pruning=pruning, verbose=verbose)

    # Don't count segments in bad-chi chunks!
    bad_chunks = (chis_chunks < corrector.chi_lo) ^ (chis_chunks > corrector.chi_hi)
    i = 0 if bad_chunks[-1] else np.where(not bad_chunks[::-1])[0][0]
    chunk_lims = list(corrector.get_chunk_lims(len(residuals), corrector.segment_size))
    last_valid_idx = np.where(corrector.block_positions >= chunk_lims[-i-1][1])[0][0]
    to_consider = np.ones(len(bestfit), dtype=bool)
    to_consider[3+len(x_knots)-(len(bestfit)-3-len(x_knots)-last_valid_idx):3+len(x_knots)] = 0
    to_consider[-(len(bestfit)-3-len(x_knots)-last_valid_idx):] = 0

    # Keep fitting and removing peaks until convergence is reached
    iteration_count = 0
    while any([x > recursion_threshold for i, x in enumerate(
            abs(new_bestfit[3:] - bestfit[3:]) / sigma) if to_consider[i+3]]):
        bestfit = np.array(new_bestfit)

        # Fit skew normals through residuals
        residuals = corrector.data.logPSD - Y_model
        popt_chunks, _, chis_chunks = corrector.fit_frequency_blocks(residuals)
        # Update peak mask
        peak_mask, _ = corrector.get_peak_mask_from_residuals(
            residuals, peak_mask, popt_chunks, chis_chunks, verbose=verbose, pruning=pruning)
        # Background fit through non-peak data
        new_bestfit, sigma, Y_model = corrector.combine_spline_slope_smart(
            x_knots, y_knots, peak_mask, bestfit[3+len(x_knots):], pruning=pruning, verbose=True)

        # Loop condition
        iteration_count += 1
        if iteration_count > 10:
            print("WARNING: Max. iteration count exceeded, breaking..")
            break
    del residuals, Y_model

    # Save results
    Y_corrected = corrector.data.logPSD - models.model_segment_slope(
        new_bestfit, np.log(corrector.data.freq), corrector.block_positions, shift=3+len(x_knots))
    # TODO: unify saving below? Might not be necessary
    with h5py.File(output_path, "w") as _f:
        dset = _f.create_dataset("logPSD", data=Y_corrected, dtype="float64")
        dset.attrs.update(corrector.data.vars.__dict__)
        _f.create_dataset("frequency", data=data.freq, dtype="float64")
        _f.create_dataset("peak_mask", data=peak_mask, dtype="bool")


if __name__ == '__main__':
    cmdl_args = parse_args()
    _data_path = cmdl_args.pop("data_path")

    if os.path.isdir(_data_path):
        prefix = cmdl_args.pop("prefix")
        for file_path in glob.glob(os.path.join(_data_path, f"{prefix}*")):
            correct_blocks(file_path, **cmdl_args)
    elif os.path.exists(_data_path):
        correct_blocks(_data_path, **cmdl_args)
    else:
        raise IOError("Invalid path: '{_data_path}'.")
