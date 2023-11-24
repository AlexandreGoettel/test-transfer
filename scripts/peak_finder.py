"""Class to find peaks in LPSD output."""
# Standard imports
import json
from tqdm import tqdm, trange
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.interpolate import interp1d, CubicSpline
from scipy.stats import norm, skewnorm
# Samplers
import ultranest
import ultranest.stepsampler
from ultranest.plot import cornerplot
import pymultinest
# Project imports
from multinest_reader import MultiNestAnalyser
import utils
import models
import hist
import sensutils


def running_median(arr, buffer):
    """Self-explanatory."""
    n = len(arr)
    result = np.zeros(n)

    for i in range(n):
        start = max(0, i-buffer)
        end = min(n, i+buffer+1)  # +1 because Python slicing is end-exclusive
        result[i] = np.median(arr[start:end])

    return result


def prior_uniform(cube, initial, epsilon):
    """Define uniform prior around initial*(1 +- epsilon)."""
    param = cube.copy()
    lo, hi = initial * (1. - epsilon), initial * (1. + epsilon)
    param = lo + param * (hi - lo)
    return param


class PeakFinder:
    """Find peaks in LPSD output."""
    def __init__(self, **kwargs):
        """Set required attributes and store output file contents as ndarray."""
        required_attrs = ["fs", "fmin", "fmax", "resolution", "epsilon", "name"]
        for name, value in kwargs.items():
            if name in required_attrs:
                required_attrs.remove(name)
                setattr(self, name, value)
            else:
                print(f"[PeakFinder] Unknown parameter '{name}', ignoring..")
        # Check if all required attributes were set
        if len(required_attrs) > 0:
            raise ValueError("Missing attributes for PeakFinder.init: " +
                             ", ".join(required_attrs))

        # Calculate useful variables
        self.J = utils.Jdes(self.fmin, self.fmax, self.resolution)
        self.g = np.log(self.fmax) - np.log(self.fmin)

        # Read data
        _, self.psd = utils.read(self.name, n_lines=self.J, raw_freq=False)
        bin_width = self.fmin * np.exp(np.arange(self.J) * self.g / float(self.J - 1)) *\
            (np.exp(self.g / float(self.J - 1)) - 1)
        self.freq = self.fmin + np.cumsum(bin_width) - bin_width[0]
        self.freq = self.freq[:len(self.psd)]

        # Protection against (old) LPSD bug
        self.psd[-1] = self.psd[-2] if self.psd[-1] == 0 else self.psd[-1]

    def block_position_gen(self):
        """Generator for block positions."""
        j0 = 0
        j = utils.j(j0, self.J, self.g, self.epsilon,
                    self.fmin, self.fmax, self.fs, self.resolution)
        yield 0  # Always start at 0
        yield int(j)

        while True:
            j0 = j
            j = utils.j(j0, self.J, self.g, self.epsilon,
                        self.fmin, self.fmax, self.fs, self.resolution)
            if j >= self.J - 1:
                yield self.J - 1
                return
            yield int(j)

    def adjust_block_positions(self, block_positions, mask):
        """Given block positions and a mask of to-be-removed indices, adjust."""
        cumulative_mask = np.cumsum(mask)
        cumulative_mask = np.append(cumulative_mask, cumulative_mask[-1])
        return cumulative_mask[block_positions] - 1

    def fit_segments(self, x, y, x_knots, y_knots,
                     block_positions, verbose=False):
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
            y_model = models.model_spline(y_knots, x_knots, shift=0)(x_segment)
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

    def simple_spline_fit(self, pruning=1, nlive=128, premade=True, verbose=False):
        """Fit simplified spline-only model."""
        x_knots = np.array([2.30258509, 3.04941017, 3.79623525,
                            8.65059825, 8.95399594, 8.97733423, 9.02401079])

        def likelihood(cube, *_):
            return models.likelihood_simple_spline(cube, np.log(self.freq[::pruning]),
                                                   np.log(self.psd[::pruning]), x_knots)

        def prior(cube, *_):
            return models.prior_simple_spline(cube, np.log(self.freq[::pruning]),
                                              np.log(self.psd[::pruning]), x_knots,
                                              premade=premade)

        parameters = [r'$\sigma$']
        for i in range(len(x_knots)):
            parameters += [f'$k_{i}$']
        out = "out/simple_"

        # Run PyMultinest
        pymultinest.run(likelihood, prior, len(parameters), n_live_points=nlive,
                        resume=False, verbose=True, outputfiles_basename=out,
                        sampling_efficiency="parameter")
        json.dump(parameters, open(out + "params.json", "w"))

        # Analyse output & plot results
        ana = MultiNestAnalyser(out)
        mean, sigma, bf = ana.getResults()
        if not verbose:
            return x_knots, bf[1:], sigma[1:], bf[0]

        print("Best fit:", bf)
        print("Mean:", mean)
        print("Sigma:", sigma)
        ax = plt.subplot(111)
        ax.plot(self.freq[::pruning], self.psd[::pruning], label="Data", zorder=1)
        x_plot = np.logspace(np.log10(self.freq[0]), np.log10(self.freq[-1]), 1000)
        ax.plot(x_plot, np.exp(models.model_spline(bf, x_knots)(np.log(x_plot))),
                label="Best fit", zorder=2)
        ax.scatter(np.exp(x_knots), np.exp(bf[1:len(x_knots)+1]),
                color="r", zorder=3)

        # Nice things
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend(loc="best")
        plt.show()

        return x_knots, bf[1:], sigma[1:], bf[0]

    def interference_fit(self, x_knots, y_knots, y_sigma, sigma,
                         block_positions, pruning, verbose=False):
        """Combined fit where the line optimisations are semi-separate."""
        X, Y = np.log(self.freq[::pruning]), np.log(self.psd[::pruning])
        _block_positions = np.array([pos / pruning for pos in block_positions], dtype=int)

        def likelihood(cube, *_):
            return models.likelihood_interference(
                cube, X, Y, sigma, x_knots, _block_positions, self.fit_segments)

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

        # Get UltraNest results
        results = sampler.run(viz_callback=False)
        sampler.print_results()
        bf = results["maximum_likelihood"]["point"]
        bf_model = models.model_interference(
            bf, X, Y, _block_positions, x_knots, self.fit_segments)
        # Also get best-fit line parameters
        popt, pcov = self.fit_segments(X, Y, x_knots, y_knots,
                                       _block_positions, verbose=False)
        if not verbose:
            return bf, popt, pcov

        # Plot results
        ax = plt.subplot(111)
        ax.plot(self.freq, self.psd, label="Data", zorder=1)

        # Plot combined model
        x_plot = np.logspace(np.log10(self.freq[0]), np.log10(self.freq[-1]), 1000)
        y_model = models.model_spline(y_knots, x_knots, shift=0)(np.log(x_plot))
        popt, _ = self.fit_segments(np.log(self.freq), np.log(self.psd), x_knots, y_knots,
                                    block_positions, verbose=False)
        pruning = int(len(self.freq) / len(x_plot))
        _block_positions = np.array([pos / pruning for pos in block_positions], dtype=int)
        y_model += models.model_segment_line(
            popt.flatten(), np.log(x_plot), _block_positions, shift=0)
        ax.plot(x_plot, np.exp(y_model), label="Interference model")

        # Nice things
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend(loc="upper right")
        return bf, popt, pcov

    def do_skew_norm_fit(self, data, bin_factor=.02,
                         cfd_factor=.05, verbose=False):
        """'Zoom' and fit around peak using a small CFD to avoid being disturbed by outliers."""
        bins = utils.apply_cfd(
            data,
            np.linspace(min(data), max(data), int(len(data)*bin_factor)),
            cfd_factor)

        # Estimate starting parameters
        h0, bins = np.histogram(data, bins)
        bin_centers = (bins[1:] + bins[:-1]) / 2.
        peak_index = np.argmax(h0)
        p0 = [1, bins[peak_index], np.sqrt(utils.seq_var(bin_centers, h0)), 0]
        p0[0] = h0[peak_index] / (hist.skew_gaus(bins[peak_index], *p0) *\
            (bins[1] - bins[0]))

        # Perform fit
        try:
            popt, pcov, chi = hist.fit_hist(hist.skew_gaus, data, bins,
                                            p0=p0, get_chi_sqr=True)
        except (ValueError, RuntimeError):
            return np.zeros_like(p0), np.zeros((len(p0), len(p0))), np.inf

        if verbose:
            ax, axRes = hist.plot_func_hist(hist.skew_gaus, popt, data, bins,
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

    def fit_frequency_blocks(self, data, cfd_alpha=.05, segment_size=10000,
                             chi_lo=.1, chi_hi=2, verbose=False):
        """Separate data in blocks and fit each block's projection with a skew norm."""
        start = 0
        chis = np.zeros(int(len(data) / segment_size)+1)
        popt, pcov = np.zeros((chis.shape[0], 4)), np.zeros((chis.shape[0], 4, 4))
        for i, (start, end) in enumerate(tqdm(self.get_chunk_lims(len(data), segment_size),
                                              total=len(chis), desc="Skew norm fits..")):
            # Perform fit
            _data = data[start:end]
            popt[i, :], pcov[i, ...], chis[i] = self.do_skew_norm_fit(
                _data, bin_factor=.01, cfd_factor=cfd_alpha, verbose=verbose)
            if chis[i] > chi_hi:
                popt[i, :], pcov[i, ...], chis[i] = self.do_skew_norm_fit(
                    _data, bin_factor=.03, cfd_factor=2*cfd_alpha, verbose=verbose)
            if chis[i] < chi_lo:
                popt[i, :], pcov[i, ...], chis[i] = self.do_skew_norm_fit(
                    _data, bin_factor=.01, cfd_factor=2*cfd_alpha, verbose=verbose)

        return popt, pcov, chis

    def get_peak_info(self, data, mask):
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

    def get_peak_mask_from_residuals(self, residuals, peak_mask, popt, chis,
                                     segment_size=1000, chi_lo=.1, chi_hi=2,
                                     cut_alpha=0.99, pruning=100, verbose=False):
        """Whiten data using skewnorm fit results and apply cut to find peaks."""
        # Filter good fits with a cut on chi^2/ndof
        goodFit = (chis > chi_lo) & (chis < chi_hi)
        # Interpolate fit results to whiten all data
        firstValid, lastValid = np.where(goodFit)[0][0], np.where(goodFit)[0][-1]
        interp_funcs = []
        for i in range(popt.shape[1] - 1):
            interp_funcs += [interp1d(
                np.arange(chis.shape[0])[goodFit],
                popt[:, i+1][goodFit]
                )]

        # ATMPTMPTMPTMTTP
        # Perform whitening on full valid data to get peak info
        # even for pre-processed peaks
        # So 1- get chunk_limits from "pre-filtered"
        # 2- translate back through peak_mask before applying

        # Whitening step - must replicate the looping from the fit
        plot_x, plot_y = [], []
        _residuals, _freq = residuals[~peak_mask], self.freq[~peak_mask]
        whitened_data = np.zeros_like(residuals)
        chunk_limits = list(self.get_chunk_lims(len(_residuals), segment_size))

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
                utils.skew_normal_cdf(residuals[_start:_end], mu=mu, sigma=sigma, alpha=alpha)
            )
            # Prepare plot
            plot_x.extend(self.freq[_start:_end])
            plot_y.extend(whitened_data[_start:_end])

        # Get positive peak information for later DM searches.
        cut_value = -norm.ppf((1. - cut_alpha) / 2.)
        peak_info = self.get_peak_info(whitened_data, whitened_data >= cut_value)

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
            ax.plot(self.freq[::pruning], self.psd[::pruning], label="Original")
            ax.plot(self.freq[~_full_mask][::pruning],
                    self.psd[~_full_mask][::pruning], label="Non-peak")
            ax.legend(loc="upper right")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD")
            plt.show()
        return _full_mask, peak_info

    def combined_fit(self, peak_mask, block_positions,
                     x_knots, y_knots, y_sigma,
                     popt_segments, pcov_segments, pruning=100):
        """Performed full combined spline + line segments fit on LPSD output."""
        X, Y = np.log(self.freq[~peak_mask][::pruning]), np.log(self.psd[~peak_mask][::pruning])
        block_positions = self.adjust_block_positions(block_positions, ~peak_mask)
        block_positions = np.array([pos / pruning for pos in block_positions], dtype=int)

        def likelihood(cube, *_):
            return models.likelihood_combined_vec(cube, X, Y, x_knots, block_positions)

        def prior(cube, *_):
            return models.prior_combined_vec(cube, y_knots, y_sigma,
                                             popt_segments, pcov_segments)

        parameters = []
        for i in range(len(x_knots)):
            parameters += [f'$k_{i}$']
        for i in range(popt_segments.shape[0]):
            for j in range(popt_segments.shape[1]):
                parameters += [f'{utils.get_nth_letter(j+1)}_{i}']

        # Run UltraNest
        sampler = ultranest.ReactiveNestedSampler(
            parameters, likelihood, prior,
            log_dir="out_para/run2/", resume="resume",
            vectorized=True, ndraw_min=512
        )
        sampler.stepsampler = ultranest.stepsampler.SliceSampler(
            nsteps=2*len(parameters),
            generate_direction=ultranest.stepsampler.generate_mixture_random_direction
        )
        results = sampler.run(viz_callback=False, frac_remain=1e-5,
                              min_num_live_points=2048)
        sampler.print_results()
        bf = results["maximum_likelihood"]["point"]
        print("BEST FIT:", bf)
        return bf

    def combined_fit_minimize(self, peak_mask, block_positions, pruning,
                              x_knots, initial_guess):
        """Maximise full combined likelihood."""
        def likelihood(cube):
            return -models.likelihood_combined(cube, X, Y, x_knots, block_positions)

        X, Y = np.log(self.freq[~peak_mask][::pruning]), np.log(self.psd[~peak_mask][::pruning])
        block_positions = self.adjust_block_positions(block_positions, ~peak_mask)
        block_positions = np.array([pos / pruning for pos in block_positions], dtype=int)

        result = minimize(likelihood, initial_guess, method="BFGS", bounds=None)
        return result.x

    def line_only_fit(self, x_knots, y_knots, block_positions,
                      peak_mask, pruning=1000, verbose=False):
        """Fit only line slopes on existing spline fit data."""
        X, Y = np.log(self.freq[~peak_mask][::pruning]), np.log(self.psd[~peak_mask][::pruning])
        block_positions = self.adjust_block_positions(block_positions, ~peak_mask)
        block_positions = np.array([pos / pruning for pos in block_positions], dtype=int)

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
                # offset = np.median(y_segment[:buffer] - y_model[:buffer]) - a*x_segment[0]
                offset = -a*x_segment[0]
                return slope*x + offset

            try:
                popt[i], pcov[i, ...] = curve_fit(
                    line, x_segment, y_segment - y_model, p0=[a], absolute_sigma=False)
            except ValueError:
                tqdm.write(f"Skipping fit #{i+1}..")

            if verbose:
                y_spline[start_pos:end_pos] +=\
                    popt[i]*x_segment - popt[i]*x_segment[0]

        if verbose:
            plt.figure()
            plt.plot(X, Y, label="Data")
            plt.plot(X, y_spline, label="Fit")
            plt.plot(X, models.model_spline(y_knots, x_knots, shift=0, extrapolate=True)(X),
                     label="Spline only")
            plt.legend(loc="upper right")
            plt.show()
        return popt, pcov

    def combine_spline_slope_smart(self, x_knots, y_knots, block_positions,
                                   popt, peak_mask,
                                   pruning=1000, verbose=False):
        """Combine spline and slope fit based on initial_guess."""
        # Prepare data
        X, Y = np.log(self.freq[~peak_mask][::pruning]), np.log(self.psd[~peak_mask][::pruning])
        block_positions = self.adjust_block_positions(block_positions, ~peak_mask)
        block_positions = np.array([pos / pruning for pos in block_positions], dtype=int)
        # mask = (X >= min(x_knots)) & (X <= max(x_knots))
        # X, Y = X[mask], Y[mask]
        X_sqr = X**2
        del peak_mask

        # Optimise
        def lkl(cube, *_):
            y_spline = models.model_spline(cube, x_knots, shift=3)(X)
            y_lines = models.model_segment_slope(cube, X, block_positions,
                                                 shift=3+len(x_knots))
            lkl = norm.logpdf(Y - y_spline - y_lines,
                              scale=1e-4 + np.abs(cube[0]*X_sqr + cube[1]*X + cube[2]))
            return lkl.sum()

        initial_guess = np.concatenate([[0, 0, 1], y_knots, popt])
        tqdm.write("Starting minimisation for combined spline+slopes fit..")
        popt_combined = minimize(lambda x: -lkl(x), initial_guess,
                                #  method="Nelder-Mead", options={"maxiter": int(2e5)})
                                 method="Powell")
        tqdm.write("Done!")
        try:
            assert popt_combined.success
        except AssertionError as e:
            print(popt_combined)
            raise e
        bf = popt_combined.x

        if verbose:
            print(popt_combined)
            plt.figure()
            plt.plot(X, Y, label="Data")
            y_spline = models.model_spline(bf, x_knots, shift=3)(X)
            y_lines = models.model_segment_slope(bf, X, block_positions,
                                                 shift=3+len(x_knots))
            plt.plot(X, y_spline, label="Spline only")
            plt.plot(X, y_spline + y_lines, label="Best-fit")
            plt.legend(loc="upper right")
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
            ax.legend(loc="upper right")
            plt.tight_layout()

            plt.figure()
            plt.plot(X, bf[0]*X_sqr + bf[1]*X + bf[2])
            plt.title("Gaussian sigma vs frequency")

            plt.show()

        # X = np.log(self.freq[self.freq < np.exp(max(x_knots))])
        X = np.log(self.freq)
        y_model = models.model_spline(bf, x_knots, shift=3)(X)
        y_model += models.model_segment_slope(bf, X, block_positions, shift=3+len(x_knots))
        return bf, y_model
