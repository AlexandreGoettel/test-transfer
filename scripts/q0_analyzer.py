"""File to find significant candidates in DM q0 data by comparing MC and data distributions."""
import os
from multiprocessing import Pool
from tqdm import trange, tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm, skewnorm
from scipy.optimize import minimize
# Project imports
import hist
import utils
import sensutils
import models


def extract_clean_q0(q0_data):
    """Clean q0 data to get correct q0."""
    pos_mu = q0_data[:, 3] > 0
    q0 = np.zeros(q0_data.shape[0])
    q0[pos_mu] = -2*(q0_data[:,   1][pos_mu] - q0_data[:, 2][pos_mu])

    # zero-lkl glitch correction
    q0[np.isinf(q0)] = max(q0[~np.isinf(q0)])
    q0[np.isinf(q0_data[:, 1])] = max(q0)
    q0[q0_data[:, 1] == 0] = max(q0)
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


def get_chi_sqr(x, y, y_spline, distr_popt, nbins=75):
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
    if alpha < -20:
        alpha = -5

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
    chi_sqr = get_chi_sqr(x, y, y_spline, distr_popt, nbins)

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
    # Data loop
    # for q0_path, lbl in zip(["singlebin_MC_noise_q0_data.npy",
    #                          "singlebin_data_q0_data.npy",
    #                          "singlebin_inj_17_q0_data.npy"],
    #                         ["noise", "data", "injected_17"]):
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
            indices.append(np.where(mask)[0])
            positions.append(x[mask])
            heights.append(y[mask])
            Zs.append(whitened_data[mask])
    return indices, positions, heights, Zs


def clusterize_candidates(idx, freqs, heights, sigmas):
    """Cluster groups of freq neighbours."""
    _freqs, _heights, _sigmas = [], [], []
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

                _heights.append(_height_values[max_index])
                _freqs.append(freqs[i][max_index])
                _sigmas.append(sigmas[i][max_index])
    return _freqs, _heights, _sigmas


def main(q0_data_path, prefix, do_hybrid=False, dfdir="endgame",
         **q0_candidate_kwargs):
    """Coordinate analysis."""
    # Fit blocks
    if do_hybrid:
        hybrid_dataloader(q0_data_path, prefix,
                          dfdir=dfdir, plotdir="endgame/plots")

    # Look for peaks in q0 landscape
    q0_candidate_kwargs.update({"dfdir": dfdir,
                                "q0_data_path": q0_data_path,
                                "prefix": prefix})
    idx, freqs, heights, sigmas = get_q0_candidates(**q0_candidate_kwargs)
    n_candidates = 0
    for freq_group in freqs:
        if len(freq_group):
            n_candidates += len(freq_group)
    print(f"Found {n_candidates} candidates!")

    # Clusterize
    freqs, heights, sigmas = clusterize_candidates(idx, freqs, heights, sigmas)
    print(f"After clusterizing: {len(freqs)} candidates remain.")

    # For each candidate, simulate raw q0 distribution assuming mu=0
    # (to get significance) and compare with asymptotic scenario
    for frequency, height in zip(freqs, heights):
        # Get data around that peak
        # Setup q0 calculation
        # Decide on Nsim
        # Simulate & hist
        # Compare with asymptotics
        pass



if __name__ == '__main__':
    kwargs = {"dfdir": "endgame",
              "threshold_sigma": 5,
              "verbose": False}
    # main("singlebin_MC_noise_q0_data.npy", "noise", **kwargs)
    main("singlebin_data_q0_data.npy", "data", **kwargs)
