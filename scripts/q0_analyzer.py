"""File to find significant candidates in DM q0 data by comparing MC and data distributions."""
import os
import sys
import glob
from multiprocessing import Pool
import h5py
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm, skewnorm
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy import constants
# Project imports
import hist
import utils
import sensutils
import models


def log_likelihood(params, Y, bkg, peak_norm, peak_shape, model_args):
    """Likelihood of finding dark matter in the data."""
    # Actual likelihood calculation
    mu_DM = params
    assert Y.shape[0] == bkg.shape[0] and Y.shape[1] == peak_shape.shape[0]
    try:
        residuals = Y - np.log(np.exp(bkg) + peak_norm*peak_shape*mu_DM)
    except Exception as err:
        raise err
    log_lkl = sensutils.logpdf_skewnorm(residuals, *[model_args[:, i][:, None] for i in range(3)])
    assert log_lkl.shape == Y.shape

    # Add infinity protection - breaks norm but ok if empirical tests
    mask = np.isinf(log_lkl)
    if np.sum(mask):
        row_mins = np.min(np.where(mask, np.inf, log_lkl), axis=1)
        log_lkl[mask] = np.take(row_mins, np.where(mask)[0])

    return np.sum(np.sum(log_lkl, axis=1))


class PeakShape(np.ndarray):
    """Hold peak shape (varies over freq.)"""

    def __new__(cls, f, path, dtype=float, buffer=None, offset=0, strides=None, order=None):
        peak_arrays = np.load(path)
        shape = (peak_arrays["data"].shape[1],)
        frequency_bounds = peak_arrays["bounds"]

        obj = super(PeakShape, cls).__new__(cls, shape, dtype, buffer, offset, strides, order)
        obj.f = np.array([f])
        obj.frequency_bounds = frequency_bounds
        obj.peak_shapes = peak_arrays["data"]
        obj._update_array()
        return obj

    def _update_array(self):
        """Update the array based on the value of f."""
        condition = np.where(self.f > self.frequency_bounds)[0]
        idx = 0 if condition.size == 0 else condition[-1] + 1
        np.copyto(self, self.peak_shapes[idx, :])

    def update_freq(self, f):
        """Update the frequency value (and as such the peak shape array if necessary)."""
        self.f = np.array([f])
        self._update_array()


class DataManager:
    """Manager LPSD data for candidate search."""

    def __init__(self, data_path=None, json_path=None, tf_dir_path=None,
                 peak_shape_path=None, data_prefix="result"):
        """Open HDF files and prepare working at different freqs."""
        self.rho_local = 0.4 / (constants.hbar / constants.e
                                * 1e-7 * constants.c)**3  # GeV/cm^3 to GeV^4

        self.peak_shape = PeakShape(0, peak_shape_path)
        self.len_peak = len(self.peak_shape)
        self.transfer_functions, self.f_A_star = self.get_f_A_star(tf_dir_path)
        self.data_info = self.get_info_from_data_path(data_path, json_path,
                                                      prefix=data_prefix)

    def __del__(self):
        """Close HDFs properly."""
        if hasattr(self, "data_info"):
            for dfile, _, _ in self.data_info:
                dfile.close()

    def parse_ifo(self, i):
        """Parse the ifo from the dset attr at index i."""
        return self.dset.attrs[str(i)].split("_")[-1]

    def get_info_from_data_path(self, data_path, json_path, prefix="result"):
        """Get all possible info from result files in data_path and json_path."""
        data_paths = sorted(list(glob.glob(os.path.join(data_path, f"{prefix}*"))))
        data_info = []
        for _data_path in tqdm(data_paths, desc="Opening data HDFs", leave=False):
            _data_info = []
            _data_info.append(h5py.File(sensutils.get_corrected_path(_data_path), "r"))
            ifo = sensutils.parse_ifo(_data_path)
            _data_info.append(ifo)
            df_key = "splines_" + sensutils.get_df_key(_data_path)
            _data_info.append(sensutils.get_results(df_key, json_path))
            assert _data_info[-1] is not None
            # _data_info.append(df_key)
            data_info.append(_data_info)

        return data_info

    def get_f_A_star(self, tf_dir):
        """Get dicts of transfer function info."""
        transfer_functions = {}
        transfer_functions["H1"] = pd.read_csv(os.path.join(tf_dir, "Amp_Cal_LHO.txt"),
                                            delimiter="\t")
        transfer_functions["L1"] = pd.read_csv(os.path.join(tf_dir, "Amp_Cal_LLO.txt"),
                                            delimiter="\t")
        f_A_star = {"H1": interp1d(transfer_functions["H1"]["Freq_o"],
                                transfer_functions["H1"]["Amp_Cal_LHO"]),
                    "L1": interp1d(transfer_functions["L1"]["Freq_Cal"],
                                transfer_functions["L1"]["amp_cal_LLO"])}
        return transfer_functions, f_A_star

    def simulate_q0_at_freq(self, test_freq, Nsim,
                            batch_size=10000, n_processes=1):
        """Simulate q0 assuming mu=0."""
        # Need bkg, model_args, peak_norm, peak_shape
        df_columns = ["x_knots", "y_knots", "alpha_skew", "loc_skew", "sigma_skew"]
        freq_Hz, bkg, beta = [np.zeros((len(self.data_info), self.len_peak))
                              for _ in range(3)]
        model_args = np.zeros((len(self.data_info), 3))
        for i, (hdf_path, ifo, df) in enumerate(self.data_info):  # For each segment
            # Find the df entry that contains the tested frequency
            mask = (df['fmin'] <= test_freq) & (df['fmax'] >= test_freq)
            # Skip this entry if test_freq is out of bounds
            if np.sum(np.array(mask, dtype=int)) == 0:
                continue
            x_knots, y_knots, alpha_skew, loc_skew, sigma_skew\
                = df[mask][df_columns].iloc[0]

            # Find closest matching frequency
            frequencies = hdf_path["frequency"]
            frequency_idx = sensutils.binary_search(frequencies, test_freq)

            # Gather relevant data
            freq_Hz[i, :] = frequencies[frequency_idx:frequency_idx+self.len_peak]
            model_args[i, :] = [alpha_skew, loc_skew, sigma_skew]

            # Construct background
            bkg[i, :] = models.model_xy_spline(
                np.concatenate([x_knots, y_knots]), extrapolate=True)(
                    np.log(frequencies[frequency_idx:frequency_idx+self.len_peak])
            )

            # Set transfer function data
            A_star_sqr = self.f_A_star[ifo](freq_Hz[i, :])**2
            beta[i, :] =\
                self.rho_local / (np.pi*freq_Hz[i, :]**3 *
                                  A_star_sqr * (constants.e / constants.h)**2)

        # Simulate following model args and calculate q0
        def make_args():
            endpoints = np.arange(0, Nsim, batch_size)
            if endpoints[-1] != Nsim:
                endpoints = np.concatenate([endpoints, [Nsim]])
            for j, (start, end) in enumerate(zip(endpoints[:-1], endpoints[1:])):
                yield j, end - start, bkg, beta, model_args, self.peak_shape

        results = []
        with Pool(n_processes, maxtasksperchild=100) as pool:
            with tqdm(total=Nsim, desc="Calculate q0", position=1, leave=False) as pbar:
                for result in pool.imap_unordered(get_q0_batch, make_args()):
                    results.append(result)
                    pbar.update(len(result[1]))

        # Merge results
        q0_data = np.zeros((Nsim, 4))
        for [i, _q0] in results:
            q0_data[i:i+_q0.shape[0], :] = _q0
        return q0_data


def get_q0_batch(args):
    """Calculate N q0 values - formatted for multiprocessing."""
    i, N, bkg, beta, model_args, peak_shape = args
    # To mimic input format
    q0_data = np.zeros((N, 4))

    # Generate background
    alpha, mu, sigma = [np.repeat(model_args[:, j][:, None], N, axis=1) for j in range(3)]
    MC_bkg = bkg + skewnorm.rvs(alpha, loc=mu, scale=sigma, size=(bkg.shape[0], N))

    # Maximise lkl
    def log_lkl(params, Y):
        return log_likelihood(params[0], Y[:, None], bkg, beta, peak_shape, model_args)

    # We can assume that most mu values in one batch will be similar
    # So get the initial guess only once, then use it for all
    test_mus = np.logspace(-40, -32, 1000)
    idx, mask = 0, np.zeros(0, dtype=bool)
    while not any(~mask):
        test_lkl = np.array([-log_lkl([mu], MC_bkg[:, idx]) for mu in test_mus])
        mask = np.isnan(test_lkl) | np.isinf(test_lkl)
        idx += 1
        if idx == N:
            return i, q0_data * np.nan
    initial_guess = test_mus[np.argmin(test_lkl[~mask])]

    # Minimize here
    for j in range(max(0, idx-1), N):
        # Can't start the minimize in a nan state
        initial_lkl  = log_lkl([initial_guess], MC_bkg[:, j])
        if np.isnan(initial_lkl) or np.isinf(initial_lkl):
            test_lkl = np.array([-log_lkl([mu], MC_bkg[:, j]) for mu in test_mus])
            mask = np.isnan(test_lkl) | np.isinf(test_lkl)
            initial_guess = test_mus[~mask][np.argmin(test_lkl[~mask])]

        popt = minimize(lambda x: -log_lkl(x, MC_bkg[:, j]),
                        [initial_guess], bounds=[(0, None)],
                        method="Nelder-Mead", tol=1e-10)

        # Get max & zero lkl
        zero_lkl = log_lkl([0], MC_bkg[:, j])
        max_lkl = -popt.fun
        if popt.x[0] > 0:
            q0_data[j, 1:] = [zero_lkl, max_lkl, popt.x[0]]

    return i, q0_data


def extract_clean_q0(q0_data):
    """Clean q0 data to get correct q0."""
    pos_mu = q0_data[:, 3] > 0
    q0 = np.zeros(q0_data.shape[0])
    q0[pos_mu] = -2*(q0_data[:,   1][pos_mu] - q0_data[:, 2][pos_mu])

    # zero-lkl glitch correction
    q0[np.isinf(q0)] = max(q0[~np.isinf(q0)])
    q0[np.isinf(q0_data[:, 1])] = max(q0)
    # q0[q0_data[:, 1] == 0] = max(q0)
    q0[q0 < 0] = 0
    return q0


def f_q0(q0, A, mu_prime_over_sigma):
    """Test asymptotic q0 distribution."""
    out = 1. / np.sqrt(8*np.pi*q0) * np.exp(-0.5*(np.sqrt(q0) - mu_prime_over_sigma)**2)
    # out[q0 == 0] = 1 - norm.cdf(mu_prime / sigma)
    w = q0[1] - q0[0]
    out[q0 - w <= 0] = 1 - norm.cdf(mu_prime_over_sigma)
    return A * out


def f_t_mu(t_mu, A, ratio):
    """Test asymptotic t_mu distribution."""
    term_left = np.exp(-0.5*(np.sqrt(t_mu) + ratio)**2)
    term_right = np.exp(-0.5*(np.sqrt(t_mu) - ratio)**2)
    return A / np.sqrt(8*np.pi*t_mu) * (term_left + term_right)


def gaus(x, A, mu, sigma):
    """Analytic gaussian."""
    return A / (np.sqrt(2*np.pi) * sigma) * np.exp(-0.5*((x - mu) / sigma)**2)


def get_y_spline(x, *params):
    """Wrap spline model on x."""
    return models.model_xy_spline(params, extrapolate=True)(x)


def get_chi_sqr(y, y_spline, distr_popt, nbins=75):
    """Calculate chi^2/dof."""
    h0, bins = np.histogram(y - y_spline, nbins)
    bin_centers = (bins[1:] + bins[:-1]) / 2.
    m = h0 != 0
    distr = np.sum(h0)*(bins[1] - bins[0])*skewnorm.pdf(
        bin_centers, distr_popt[0], loc=distr_popt[1], scale=distr_popt[2])
    return np.sum((h0[m] - distr[m])**2 / h0[m]) / (len(h0[m]) - len(distr_popt))


def get_bic(x, y, n_knots, buffer=20, nbins=100, verbose=False):
    """Perform spline fit and return skew-norm BIC."""
    # 1. Choose x_knots
    x_knots = np.linspace(min(x), max(x), n_knots)
    # 1.1 Choose y_knots based on y data
    y_knots = np.array([np.median(y[max(0, np.where(x >= knot)[0][0] - buffer):
        np.where(x >= knot)[0][0] + buffer]) for knot in x_knots])

    # 2. Bound x in non-overlapping blocks
    bounds = [(x_knots[0], (x_knots[1] + x_knots[0]) / 2)]
    epsilon = 0.001 * (bounds[0][1] - bounds[0][0])
    bounds += [(0.5 * (x_knots[i] + x_knots[i-1]) + epsilon,
                0.5 * (x_knots[i] + x_knots[i+1]) - epsilon)
                for i in range(1, len(x_knots)-1)]
    bounds += [(0.5 * (x_knots[-2] + x_knots[-1]), x_knots[-1])]
    # 3. Add y bounds
    bounds += [(None, None) for y in y_knots]

    # 4. Minimize mean square error
    def lkl(params):
        y_spline = models.model_xy_spline(params, extrapolate=True)(x)
        return ((y - y_spline)**2).sum()
    popt = minimize(lkl, np.concatenate([x_knots, y_knots]),
                    method="Nelder-Mead", bounds=bounds)

    y_spline = models.model_xy_spline(popt.x, extrapolate=True)(x)
    if verbose:
        print(popt)
        plt.plot(x, y, zorder=1)
        plt.plot(x, y_spline, zorder=2)
        plt.scatter(popt.x[:n_knots], popt.x[n_knots:], color="r", zorder=3)

    # 5. Get starting parameters for skew normal
    res = y - y_spline
    alpha, mu, sigma = sensutils.get_skewnorm_p0(res)

    # FIXME
    if np.abs(alpha) > 5:
        alpha = np.sign(alpha)*5

    # 6. Fit a skew normal
    def fitFunc(_x, *params):
        return params[3]*skewnorm.pdf(_x, params[0], loc=params[1], scale=params[2])

    h0, bins = np.histogram(res, nbins)
    try:
        _popt, _ = hist.fit_hist(fitFunc, res, bins, [alpha, mu, sigma, max(h0)])
    except RuntimeError:
        return -np.inf, popt.x, [alpha, mu, sigma]

    # 7. Calculate BIC :-)
    _lkl = skewnorm.pdf(res, _popt[0], loc=_popt[1], scale=_popt[2])
    bic = _lkl.sum() - 0.5 * (2*len(x_knots) + 4) * np.log(len(x))

    if verbose:
        ax = hist.plot_func_hist(fitFunc, _popt, res, bins)
        ax.set_title(f"BIC: {bic:.1e}")
        plt.show()

    return bic, popt.x, _popt[:-1]


def process_iteration(params):
    """Wrapper for parallel calculation."""
    i, x, y, kwargs, plot_kwargs = params
    best_fit, f_popt, distr_popt = sensutils.bayesian_regularized_linreg(x, y, **kwargs)

    # Plots
    start, end, plotdir, prefix, nbins = list(map(
        lambda x: plot_kwargs[x], ["start", "end", "plotdir", "prefix", "nbins"]))

    suffix = f"{start:.1f}_{end:.1f}.png"
    plt.savefig(os.path.join(plotdir, f"{prefix}_spline_{suffix}"))
    plt.close()
    y_spline = models.model_xy_spline(f_popt, extrapolate=True)(x)
    chi_sqr = get_chi_sqr(y, y_spline, distr_popt, nbins)

    res = y - best_fit
    bins = np.linspace(min(res), max(res), nbins)
    ax = hist.plot_hist(res, bins, density=True)
    ax.plot(bins, skewnorm.pdf(bins, distr_popt[0], loc=distr_popt[1], scale=distr_popt[2]))
    ax.set_title(f"Chi^2/dof: {chi_sqr:.1f}")
    plt.savefig(os.path.join(plotdir, f"{prefix}_skewnorm_{suffix}"))
    plt.close()
    return i, f_popt, distr_popt, chi_sqr


def hybrid_fits(freq, q0, dfdir="", plotdir="", segment_size=10000,
                prefix="", verbose=True, n_processes=6,
                k_min=4, k_pruning=1, max_plateau=5, buffer=20, nbins=75):
    """Fit spline + skew norm through data."""
    # Prepare df
    df_path = os.path.join(dfdir, f"{prefix}_q0_BIC_data.json")
    df = pd.DataFrame(columns=["fmin", "fmax", "x_knots", "y_knots",
                               "alpha_skew", "loc_skew", "sigma_skew", "chi_sqr"])

    # Get data segments in log space
    positions = np.concatenate([
        np.arange(0, len(freq) - 1, segment_size),
        [len(freq) - 1]])
    def make_args():
        for i, (start, end) in enumerate(zip(positions[:-1], positions[1:])):
            end = min(end, len(freq)-1)
            data = np.log(q0[start:end])
            mask = np.isinf(data)
            x, y = freq[start:end][~mask], data[~mask]
            kwargs = dict(get_bic=get_bic, f_fit=get_y_spline, k_min=k_min,
                          max_plateau=max_plateau, k_pruning=k_pruning, plot_mean=True,
                          kernel_size=800, verbose=verbose, buffer=buffer, nbins=nbins,
                          disable_tqdm=True)
            plot_kwargs = dict(start=freq[start], end=freq[end], plotdir=plotdir,
                               prefix=prefix, nbins=nbins)
            # best_fit, f_popt, distr_popt = sensutils.bayesian_regularized_linreg(x, y, **kwargs)
            yield i, x, y, kwargs, plot_kwargs

    # Parallel execution
    with Pool(processes=n_processes, maxtasksperchild=10) as pool:
        results = []
        with tqdm(total=len(freq) // segment_size + 1, desc="BIC") as pbar:
            for result in pool.imap_unordered(process_iteration, make_args()):
                results.append(result)
                pbar.update(1)

    # Post-processing
    zipped_positions = list(zip(positions[:-1], positions[1:]))
    for result in tqdm(results, desc="Combine results"):
        i, f_popt, distr_popt, chi_sqr = result
        start, end = zipped_positions[i]

        assert not len(f_popt) % 2
        n_knots = len(f_popt) // 2
        df = pd.concat([df, pd.DataFrame(
            {"fmin": freq[start],
             "fmax": freq[end],
             "x_knots": [list(f_popt[:n_knots])],
             "y_knots": [list(f_popt[n_knots:])],
             "alpha_skew": distr_popt[0],
             "loc_skew": distr_popt[1],
             "sigma_skew": distr_popt[2],
             "chi_sqr": chi_sqr
             })])
    df.to_json(df_path, orient="records")


def hybrid_dataloader(q0_path, lbl, verbose=False, pruning=100,
                      dfdir="endgame", plotdir="endgame/plots"):
    """Prepare data for hybrid block fits."""
    # Create q0 array
    q0_data = np.load(q0_path)
    q0 = extract_clean_q0(q0_data)

    if verbose:
        fig = plt.figure()
        gs = GridSpec(5, 4, fig)
        ax_freq = fig.add_subplot(gs[:2, :])
        ax_hist = fig.add_subplot(gs[3:, :])
        bins = np.logspace(0.1, 5, 500)

        # Plot q0 vs freq
        ax_freq.plot(q0_data[::pruning, 0][q0[::pruning] > 0],
                    np.sqrt(q0[::pruning][q0[::pruning] > 0]),
                    label=lbl)

        # Hist q0
        hist.plot_hist(np.sqrt(q0), bins, ax=ax_hist,
                    density=False, label=lbl)
        ax_hist.set_xscale("log")
        ax_hist.set_yscale("log")
        ax_hist.legend(loc="best")
        ax_freq.set_xscale("log")
        ax_freq.set_yscale("log")
        ax_freq.legend(loc="best")
        plt.show()

    # Fit blocks to identify peaks
    hybrid_fits(q0_data[:, 0], q0, dfdir, plotdir=plotdir, prefix=lbl)


def get_q0_candidates(q0_data_path=None, prefix="", dfdir="endgame",
                      threshold_sigma=5, verbose=False):
    """Create a list of peaks in q0 based on threshold from whitened data."""
    # Fit info
    df_columns = ["fmin", "fmax", "x_knots", "y_knots",
                  "alpha_skew", "loc_skew", "sigma_skew", "chi_sqr"]
    df_path = os.path.join(dfdir, f"{prefix}_q0_BIC_data.json")
    with open(df_path, "r") as _f:
        df = pd.read_json(_f)

    # q0 data
    q0_data = np.load(q0_data_path)
    q0 = extract_clean_q0(q0_data)

    indices, positions, heights, Zs = [], [], [], []
    for _, sub_df in tqdm(df.iterrows(), desc=f"Candidate search '{prefix}'", total=len(df)):
        fmin, fmax, x_knots, y_knots, alpha, loc, sigma, chi_sqr\
            = sub_df[df_columns]

        # Take residuals
        fmin_idx = sensutils.binary_search(q0_data[:, 0], fmin)
        fmax_idx = 1 + sensutils.binary_search(q0_data[:, 0], fmax)
        x = q0_data[fmin_idx:fmax_idx, 0]
        y = np.log(q0[fmin_idx:fmax_idx])

        y_spline = models.model_xy_spline(np.concatenate([x_knots, y_knots]),
                                          extrapolate=True)(x)
        residuals = y - y_spline
        clean = np.isinf(residuals)
        x, y, y_spline, residuals = x[~clean], y[~clean], y_spline[~clean], residuals[~clean]

        # Whiten
        whitened_data = norm.ppf(
            utils.skew_normal_cdf(residuals, alpha=alpha, mu=loc, sigma=sigma)
        )
        # Threshold - corrected for LEE  # FIXME Set threshold using injections?
        threshold = norm.ppf(1. - (1. - norm.cdf(threshold_sigma)) / len(y))
        # Clean
        whitened_data[np.isinf(whitened_data)] = np.sign(whitened_data[np.isinf(whitened_data)])\
            * threshold*1.0001

        # Plot
        corr_sigma = norm.ppf(1 - len(y) * (1 - norm.cdf(max(whitened_data))))
        if verbose and corr_sigma > threshold_sigma:
            fig = plt.figure(figsize=(10, 16))
            gs = GridSpec(6, 8, fig)
            ax_spline = fig.add_subplot(gs[:4, :4])
            ax_res = fig.add_subplot(gs[4:, :4])
            ax_whiten = fig.add_subplot(gs[:, 4:])

            # Plot data
            ax_spline.plot(x, y, zorder=1)
            ax_spline.plot(x, y_spline, zorder=2, color="C1")
            for candidate in x[whitened_data > threshold]:
                ax_spline.axvline(candidate, linestyle="--", color="r", linewidth=.5)

            # Plot residuals
            bins = np.linspace(min(residuals), max(residuals), 200)
            ax_res = hist.plot_hist(residuals, bins, ax=ax_res, density=True)
            bin_centers = (bins[1:] + bins[:-1]) / 2.
            ax_res.plot(bin_centers, skewnorm.pdf(bin_centers, alpha, loc=loc, scale=sigma),
                        color="C1", label=f"chi^2/ndof: {chi_sqr:.1f}")
            ax_res.legend(loc="upper left")

            # Plot whitened data
            bins = np.linspace(min(whitened_data[~np.isinf(whitened_data)]),
                               max(whitened_data[~np.isinf(whitened_data)]),
                               200)
            ax_whiten = hist.plot_hist(whitened_data, bins, ax=ax_whiten, logy=True, density=True)
            bin_centers = (bins[1:] + bins[:-1]) / 2.
            ax_whiten.plot(bin_centers, norm.pdf(bin_centers), color="C1", linewidth=2)
            ax_whiten.axvline(threshold, color="r", linestyle="--")
            ax_whiten.set_title(f"Max. corrected sigma: {corr_sigma:.1f}")
            ax_whiten.set_ylim(.5 / len(whitened_data), ax_whiten.get_ylim()[1])
            plt.show()

        mask = whitened_data >= threshold
        if np.sum(mask):
            indices.append(fmin_idx + np.where(mask)[0])
            positions.append(x[mask])
            heights.append(y[mask])
            Zs.append(whitened_data[mask])
    return indices, positions, heights, Zs


def clusterize_candidates(idx, freqs, heights, sigmas):
    """Cluster groups of freq neighbours."""
    _indices, _freqs, _heights, _sigmas = [], [], [], []
    for i, _idx in enumerate(idx):
        start_index = 0
        # Iterate through each list in idx to find continuous blocks
        for j in range(1, len(_idx) + 1):
            if j == len(_idx) or _idx[j] != _idx[j-1] + 1:
                # Process the block from start_index to j-1
                _height_values = list(heights[i])
                block_max_y = max(_height_values[start_index:j])
                max_index = _height_values.index(block_max_y, start_index, j)
                start_index = j

                _indices.append(idx[i][max_index])
                _heights.append(_height_values[max_index])
                _freqs.append(freqs[i][max_index])
                _sigmas.append(sigmas[i][max_index])
    return _indices, _freqs, _heights, _sigmas


def get_data_around_candidate(mngr, test_freq, buffer=50):
    # Get the relevant data
    df_columns = ["x_knots", "y_knots", "alpha_skew", "loc_skew", "sigma_skew"]
    Y, freq_Hz, bkg, peak_norm = [np.zeros((len(mngr.data_info), int(mngr.len_peak+2*buffer)))
                                  for _ in range(4)]
    ifos = [""]*len(mngr.data_info)
    model_args = np.zeros((len(mngr.data_info), int(mngr.len_peak+2*buffer), 3))
    for i, (hdf_path, ifo, df) in enumerate(mngr.data_info):  # For each segment
        # Find the df entry that contains the tested frequency
        mask = (df['fmin'] <= test_freq) & (df['fmax'] >= test_freq)
        # Skip this entry if test_freq is out of bounds
        if np.sum(np.array(mask, dtype=int)) == 0:
            continue
        x_knots, y_knots, alpha_skew, loc_skew, sigma_skew = df[mask][df_columns].iloc[0]
        ifos[i] = ifo

        # Find closest matching frequency
        frequencies = hdf_path["frequency"]
        f_id = sensutils.binary_search(frequencies, test_freq)
        imin, imax = f_id - buffer, f_id + buffer + mngr.len_peak

        # Gather relevant data
        Y[i, :] = hdf_path["logPSD"][imin:imax]
        freq_Hz[i, :] = frequencies[imin:imax]
        model_args[i, ...] = np.array([alpha_skew, loc_skew, sigma_skew])

        # Construct background
        bkg[i, :] = models.model_xy_spline(
            np.concatenate([x_knots, y_knots]), extrapolate=True)(
                np.log(frequencies[imin:imax])
        )

        # Set transfer function data
        A_star_sqr = mngr.f_A_star[ifo](freq_Hz[i, :])**2
        peak_norm[i, :] =\
            mngr.rho_local / (np.pi*freq_Hz[i, :]**3 *
                              A_star_sqr * (constants.e / constants.h)**2)
    mngr.peak_shape.update_freq(test_freq)
    return Y, freq_Hz, bkg, peak_norm, ifos, np.array(mngr.peak_shape[:]), model_args


def plot_candidate(mu, Y, freq_Hz, bkg, peak_norm, ifos, peak_shape,
                   *_, buffer=50):
    """Do a candidate-style plot for H1 and L1."""
    # Prepare the plots
    fig = plt.figure()
    gs = GridSpec(1, 2, figure=fig)
    axH1 = fig.add_subplot(gs[0, 0])  # First row, first column
    axL1 = fig.add_subplot(gs[0, 1], sharex=axH1, sharey=axH1)

    # Plot the PSDs
    for i in range(Y.shape[0]):
        if ifos[i] == "":
            continue
        ax = axH1 if ifos[i] == "H1" else axL1
        ax.plot(freq_Hz[i, :], np.exp(Y[i, :]), alpha=.5, zorder=1)
        ax.plot(freq_Hz[i, :], np.exp(bkg[i, :]), alpha=.5, color="C0", zorder=0)

    # Find the first H1 and L1 peak norm terms
    peak_norm_H1 = peak_norm[ifos.index("H1"), :] if "H1" in ifos else 0
    peak_norm_L1 = peak_norm[ifos.index("L1"), :] if "L1" in ifos else 0

    len_peak, peak_shape[peak_shape == 0] = len(peak_shape), np.nan
    # Min-max plots
    for ifo, ax, _peak_norm in zip(["H1", "L1"], [axH1, axL1], [peak_norm_H1, peak_norm_L1]):
        # Plot reconstructed peaks
        ax.scatter(freq_Hz[0, buffer:buffer + len_peak],
                   peak_shape*mu*_peak_norm[buffer:buffer + len_peak],
                   color="k", zorder=2)
        # Nice things
        ax.set_title(ifo + r", $\Lambda_i^{-1}$:" +
                        f" {np.sqrt(mu):.1e}")
        ax.set_xlabel("Frequency a.u.")
        ax.grid(linestyle="--", linewidth=1, color="grey", alpha=.33)
    axL1.set_ylabel("log(PSD)")
    axL1.set_yticklabels([])
    axH1.set_yscale("log")
    return axH1, axL1


def tmp(mu, Y, freq_Hz, bkg, peak_norm, ifos, peak_shape):
    """Temporary playground to plot candidate-related info."""
    # 1. Plot N_segment_over_mu / N_segment per ifo.
    # imin, imax = f_id - buffer, f_id + buffer + mngr.len_peak
    isH1 = np.array([ifo == "H1" for ifo in ifos], dtype=bool)
    buffer = (Y.shape[1] - len(peak_shape)) // 2

    N_over_H1 = np.sum(np.exp(Y[isH1, buffer]) > mu*peak_shape*peak_norm[isH1, buffer])
    N_over_L1 = np.sum(np.exp(Y[~isH1, buffer]) > mu*peak_shape*peak_norm[~isH1, buffer])
    return N_over_H1, N_over_L1, np.sum(isH1)


def validate_q0(mngr, freqs, ref_q0s, n_processes=1,
                n_sim=10000, batch_size=300, threshold=5):
    """Simulations to cross-check the validity of candidates."""
    validated = np.zeros(len(freqs), dtype=int)
    for i, (freq, ref_q0) in enumerate(tqdm(
            zip(freqs, ref_q0s), total=len(freqs), desc="Validating q0")):
        q0_data = mngr.simulate_q0_at_freq(
            freq, n_sim, batch_size=batch_size, n_processes=n_processes)
        q0 = extract_clean_q0(q0_data)
        q0 = q0[q0 > 0]
        if len(q0) == 0:
            validated[i] = 2
        elif ref_q0 > np.median(q0) + threshold*np.std(q0):
            validated[i] = 1

        # tqdm.write(f"{validated[i]}")
        if validated[i] != 1:
            print(q0)
            print(q0_data[:, 1])
            print(q0_data[:, 2])
            print(q0_data[:, 3])
            tqdm.write("--")
    return validated


def check_ifo_consistency(Y, _, bkg, peak_norm,
                          ifos, peak_shape, model_args):
    """Reconstruct mu in both ifos, with uncertainty, and return consistency-test data."""
    isH1 = np.array([ifo == "H1" for ifo in ifos], dtype=bool)
    isL1 = np.array([not x for x in isH1], dtype=bool)
    idx = (Y.shape[1] - len(peak_shape)) // 2

    q0s, mus, sigmas = [], [], []
    for mask in isH1, isL1, np.ones(len(isH1), dtype=bool):
        # Get args
        _Y, _bkg, _beta = list(map(lambda x: x[mask, idx][:, None], [Y, bkg, peak_norm]))
        _model_args = model_args[mask, idx, :]

        # Def lkl to minimize
        def log_lkl(params):
            return log_likelihood(params, _Y, _bkg, _beta, peak_shape, _model_args)

        # Get initial guess
        test_mus = np.logspace(-40, -32, 1000)
        test_lkl = np.array([-log_lkl(mu) for mu in test_mus])
        mask = np.isnan(test_lkl) | np.isinf(test_lkl)
        if np.sum(~mask) == 0:
            mus.append(0)
            q0s.append(0)
            sigmas.append(0)
            continue
        initial_guess = test_mus[~mask][np.argmin(test_lkl[~mask])]

        # Minimize
        popt = minimize(lambda x: -log_lkl(x),
                        initial_guess, bounds=[(0, None)],
                        method="Nelder-Mead", tol=1e-10)
        zero_lkl = log_lkl(0)
        q0 = -2 * (zero_lkl + popt.fun)
        mu_hat = popt.x[0]
        if mu_hat == 0:
            mus.append(0)
            q0s.append(0)
            sigmas.append(0)
            continue

        # Get uncertainty
        dx = abs(mu_hat) / 2.
        max_iterations = 17
        for _ in range(max_iterations):
            lkl_left, lkl_right = log_lkl(mu_hat - 2*dx), log_lkl(mu_hat + 2*dx)
            if not np.isnan(lkl_left) and not np.isnan(lkl_right):
                break
            dx /= 2.

        try:
            sigma = sensutils.sigma_at_point(
                log_lkl, mu_hat,
                initial_dx=dx,
                tolerance=1e-4)
        except ValueError:
            mus.append(np.nan)
            q0s.append(np.nan)
            sigmas.append(np.nan)
            continue

        mus.append(mu_hat)
        q0s.append(q0)
        sigmas.append(sigma)
    # Test for consistency
    return q0s, mus, sigmas


def main(q0_data_path, prefix, DM_kwargs,
         do_hybrid=False, do_crosscheck=False,
         dfdir="endgame", **q0_candidate_kwargs):
    """Coordinate analysis."""
    # Fit blocks
    if do_hybrid:
        hybrid_dataloader(q0_data_path, prefix,
                          dfdir=dfdir, plotdir="endgame/plots")

    # Look for peaks in q0 landscape
    q0_candidate_kwargs.update({"dfdir": dfdir,
                                "q0_data_path": q0_data_path,
                                "prefix": prefix})
    # TODO: You'll have to think more about idx positions if len_peak > 1
    idx, freqs, heights, sigmas = get_q0_candidates(**q0_candidate_kwargs)
    n_candidates = sum([len(group) for group in freqs])
    print(f"Found {n_candidates} candidates!")

    # Clusterize
    idx, freqs, heights, sigmas = clusterize_candidates(idx, freqs, heights, sigmas)
    print(f"After clusterizing: {len(freqs)} candidates remain.")

    mngr = DataManager(**DM_kwargs)
    if do_crosscheck:
        # For each candidate, simulate raw q0 distribution assuming mu=0
        # (to get significance) and compare with asymptotic scenario
        q0_val_info = validate_q0(mngr, freqs, np.exp(heights),
                                  threshold=q0_candidate_kwargs["threshold_sigma"],
                                  n_sim=100, batch_size=10, n_processes=10)
        np.save("q0_val_info.npy", q0_val_info)
    else:
        q0_val_mask = np.load("q0_val_info.npy")

    # Apply
    # idx, freqs, heights, sigmas = list(map(lambda x: x[q0_val_mask],
    #                                        [idx, freqs, heights, sigmas]))
    # print(f"After q0 validation: {len(freqs)} candidates remain.")

    # Consistency checks
    is_consistent = np.zeros(len(freqs), dtype=bool)
    individual_q0 = np.zeros((len(freqs), 3))
    new_mus = np.zeros((len(freqs)))
    t_threshold = 3  # sigma
    for i, (frequency, log_q0val) in enumerate(tqdm(zip(freqs, heights), total=len(freqs))):
        # Check the difference in reconstructed mu between H1 and L1
        candidate_data = get_data_around_candidate(mngr, frequency)
        _q0s, _mus, _sigmas = check_ifo_consistency(*candidate_data)
        t_value = abs(_mus[0] - _mus[1]) / np.sqrt(_sigmas[0]**2 + _sigmas[1]**2)
        is_consistent[i] = not np.isnan(t_value) and\
            not np.isnan(_q0s[2]) and\
            t_value <= t_threshold

        individual_q0[i, :] = _q0s
        new_mus[i] = _mus[2]
        if is_consistent[i]:
            tqdm.write(f"Old q0:\t{np.exp(log_q0val):.1f}\tNew:\t{_q0s[2]:.1f}")

    print(f"After consistency_check: {np.sum(is_consistent)} candidates remain.")
    print(f"TMP: {np.sum(individual_q0[:, 2] >= 9)} remain significant")

    # Plot candidates
    raw_q0_data = np.load(q0_data_path)
    for new_q0s, mu, _is_consistent, _idx, frequency in tqdm(
            zip(individual_q0, new_mus, is_consistent, idx, freqs), total=len(freqs)):
        if not _is_consistent or new_q0s[2] < 9:
            continue
        # Plot candidate data
        # mu = raw_q0_data[_idx, 3]
        candidate_data = get_data_around_candidate(mngr, frequency)
        axH1, axL1 = plot_candidate(mu, *candidate_data)
        axH1.set_title(f"Total q0: {new_q0s[2]:.1f}, H1 only: {new_q0s[0]:.1f}")
        axL1.set_title(f"L1 only q0: {new_q0s[1]:.1f}")
        q0_data = mngr.simulate_q0_at_freq(
            frequency, 10000, batch_size=300, n_processes=10)
        q0 = extract_clean_q0(q0_data)
        q0 = q0[q0 > 0]
        ax = hist.plot_hist(q0, 75, logy=True)
        ax.axvline(new_q0s[2], color="r", linestyle="--")
        plt.show()


if __name__ == '__main__':
    q0_kwargs = {"dfdir": "endgame",
                 "threshold_sigma": 5,
                 "verbose": False
    }
    DM_kwargs = {"data_path": "data/",
                 "json_path": "data/processing_results.json",
                 "peak_shape_path": "single_bin_peak_shape_data.npz",
                 "tf_dir_path": "data/transfer_functions"
    }
    # main("singlebin_MC_noise_q0_data.npy", "noise", **kwargs)
    main("singlebin_data_q0_data.npy", "data", DM_kwargs,
         do_hybrid=False, do_crosscheck=False, **q0_kwargs)
