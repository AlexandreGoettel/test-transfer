"""Calculate experimental upper limits based on real data."""
from multiprocessing import Pool
from tqdm import tqdm, trange
import glob
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import constants
from scipy.optimize import minimize, newton, OptimizeResult
from scipy.stats import norm, skewnorm
from scipy.interpolate import interp1d
from scipy.special import erf
# Project imports
import sensutils
import models
import hist

# tmp
import warnings
warnings.filterwarnings("ignore")


# Global constant for peak normalisation in likelihood calculation
rho_local = 0.4 / (constants.hbar / constants.e * 1e-7 * constants.c)**3  # GeV/cm^3 to GeV^4


def log_likelihood(param, Y, bkg, peak_norm, model_args):
    """Likelihood of finding dark matter in the data!"""
    residuals = Y - np.log(np.exp(bkg) + peak_norm*param)
    lkl = sensutils.logpdf_skewnorm(
        residuals, model_args[:, 0], model_args[:, 1], model_args[:, 2])
    if len(lkl.shape) == 2:
        return np.sum(lkl, axis=1)
    elif len(lkl.shape) == 1:
        return lkl.sum()
    else:
        raise ValueError


def parse_ifo(name):
    """Get ifo from data path."""
    return name.split(".")[-2].split("/")[-1].split("_")[-1]


def binary_search(arr, value):
    """Perform a binary search to find left-side bounds on value."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        mid_val = arr[mid]
        if mid_val < value:
            lo = mid + 1
        elif mid_val > value:
            hi = mid - 1
        else:
            return mid
    return lo


def f_q_mu_tilda(q_mu, mu, mu_prime, sigma):
    """Asymptotic approximation to f(q_mu_tilda|mu_prime)."""
    out = 0.5 / np.sqrt(2*np.pi*q_mu)*np.exp(-0.5*(np.sqrt(q_mu) - (mu-mu_prime)/sigma)**2)
    mask = (mu/sigma)**2 >= q_mu
    out[mask] = 0.5 / (np.sqrt(2*np.pi)*mu/sigma) * np.exp(-0.5*((q_mu[mask] - (mu**2 - 2*mu*mu_prime)/sigma**2) / (2*mu/sigma))**2)
    return out


def cumf_q_mu(q_mu, mu, mu_prime, sigma):
    """Cumulative probability function for f(q_mu|mu_prime)."""
    out = norm.cdf(np.sqrt(q_mu) - (mu - mu_prime) / sigma)
    mask = q_mu > (mu / sigma)**2
    if len(mask):
        out[mask] = norm.cdf((q_mu[mask] - (mu**2 - 2*mu*mu_prime) / sigma**2) / (2*mu/sigma))
    return out


def profile_calc(bkg, peak_norm, model_args, x0, mu_range,
                 alpha_CL=.95, n_q_mu=47500, tol=1e-5, no_tqdm=False):
    """Full profile calculation."""
    def log_lkl(param, data):
        return log_likelihood(param, data, bkg, peak_norm, model_args)

    # Generate a bunch of q_mus by fluctuating background
    def get_q_mu(mu_test, data, mu_hat, max_lkl, zero_lkl, full=True):
        q_mu = np.zeros_like(mu_hat)
        if full:
            mu_hat = np.zeros(data.shape[0])
            max_lkl, zero_lkl = np.zeros(data.shape[0]), np.zeros(data.shape[0])
            for i, yi in tqdm(enumerate(data), desc="q_mu", position=2, leave=False,
                              total=len(data), disable=no_tqdm):
                popt = minimize(lambda x: -log_lkl(x, yi),
                                mu_test/2,
                                method="Nelder-Mead",
                                bounds=[(0, mu_range[1])],
                                tol=tol)
                if not popt.success:
                    print(mu_test/2)
                    print(mu_range)
                    print(popt)
                # assert popt.success
                mu_hat[i], max_lkl[i] = popt.x, -popt.fun
                zero_lkl[i] = log_lkl(0, yi)

        mask = mu_hat < mu_test
        # print(mu_test, mu_hat, data[mask].shape, max_lkl[mask].shape)
        q_mu[mask] = -2 * (log_lkl(mu_test, data[mask]) - max_lkl[mask])
        mask = mu_hat < 0
        q_mu[mask] = -2 * (log_lkl(mu_test, data[mask]) - zero_lkl[mask])
        return q_mu

    # n_q_mu_0, n_q_mu_mu = n_q_mu
    data_mu_0 = bkg + skewnorm.rvs(model_args[:, 0], loc=model_args[:, 1], scale=model_args[:, 2],
                                   size=(n_q_mu, len(bkg)))
    full_bkg = bkg + skewnorm.rvs(model_args[:, 0], loc=model_args[:, 1], scale=model_args[:, 2],
                                  size=(n_q_mu, len(bkg)))

    # Get mu_hat etc. for data_mu_0 and full_bkg
    def get_mu_hat_etc(data):
        mu_hat = np.zeros(data.shape[0])
        max_lkl, zero_lkl = np.zeros(data.shape[0]), np.zeros(data.shape[0])
        for i, yi in tqdm(enumerate(data), desc="q_mu", position=2, leave=False,
                          total=len(data), disable=no_tqdm):
            popt = minimize(lambda x: -log_lkl(x, yi),
                            mu_range[1]/2,
                            method="Nelder-Mead",
                            bounds=[(0, mu_range[1])],
                            tol=tol)
            assert popt.success
            mu_hat[i], max_lkl[i] = popt.x, -popt.fun
            zero_lkl[i] = log_lkl(0, yi)
        return mu_hat, max_lkl, zero_lkl

    try:
        mu_hat_0, max_lkl_0, zero_lkl_0 = get_mu_hat_etc(data_mu_0)
        mu_hat_mu_0, max_lkl_mu_0, zero_lkl_mu_0 = get_mu_hat_etc(full_bkg)
    except AssertionError:
        return np.nan

    def get_p_val(mu_test, verbose=False):
        q_mu_0 = get_q_mu(mu_test, data_mu_0, mu_hat_0, max_lkl_0, zero_lkl_0,
                          full=False)
        data_mu_mu = np.log(np.exp(full_bkg) + peak_norm*mu_test)
        q_mu_mu = get_q_mu(mu_test, data_mu_mu, mu_hat_mu_0, max_lkl_mu_0, zero_lkl_mu_0,
                           full=True)

        # Calculate p-value
        med_q_mu_0 = np.median(q_mu_0)
        k = np.sum(np.array(q_mu_mu >= med_q_mu_0, dtype=int))
        p_val, p_sigma = k / float(n_q_mu), np.sqrt(sensutils.get_eff_var(k, n_q_mu))
        if type(mu_test) == type(np.array([])) or type(mu_test) == list:
            mu_test = mu_test[0]
        if verbose:
            bins = np.linspace(0, max(q_mu_0), 100)
            ax = hist.plot_hist(q_mu_0, bins, logy=True, label="f(q_mu|0)", density=True)
            ax = hist.plot_hist(q_mu_mu, bins, ax=ax, label="f(q_mu|mu)", density=True)
            ax.axvline(med_q_mu_0, linestyle="--", color="r")
            ax.set_title(f"{mu_test:.1e}: {p_val:.3f} +- {p_sigma:.4f}")
            plt.show()
        return (p_val - (1 - alpha_CL))**2

    # Find good starting value
    x0 = .5*x0[0]
    while np.allclose(get_p_val(x0), (1-alpha_CL)**2):
        tqdm.write(f"Adjusting x0 {x0:.1e}")
        x0 /= 2
    # Minimize: get mu for which p-val is correct
    popt = minimize(get_p_val, x0,
                    bounds=[mu_range],
                    method="Nelder-Mead",
                    tol=1e-5)
    # get_p_val(popt.x, verbose=True)
    return popt


def find_leading_trailing_invalid(arr):
    # Create a boolean mask where True indicates either NaN or Inf
    mask = np.isnan(arr) | np.isinf(arr)
    if not np.sum(~mask):
        return 0, 0

    # Find the last index of the leading group of invalid numbers
    if mask[0]:
        for i, val in enumerate(mask):
            if not val:
                first_idx = i - 1
                break
    else:
        first_idx = 0

    # Find the first index of the trailing group of invalid numbers
    if mask[-1]:
        for i, val in enumerate(mask[::-1]):
            if not val:
                last_idx = len(arr) - 1 - (i - 1)
                break
    else:
        last_idx = len(arr) - 1

    return first_idx + 1, last_idx - 1


def get_valid_mu_ranges(Y, bkg, peak_norm, model_args,
                        start=-40, end=-30, n=100):
    """Investigate where mu is valid for an easier later analysis."""
    mu = np.logspace(start, end, n)
    output = []
    def get_mu_ranges(test_mus):
        valid_mu_ranges = np.zeros((len(Y), 2))
        for i, logPSD in enumerate(Y):
            residuals = logPSD - np.log(np.exp(bkg[i]) + peak_norm[i]*test_mus)
            log_lkl_values = sensutils.logpdf_skewnorm(residuals, *model_args[i])
            start, end = find_leading_trailing_invalid(log_lkl_values)

            if not np.isnan(log_lkl_values[0]) and start:
                start -= 1
            assert not np.sum(np.isnan(log_lkl_values[start:end]))

            valid_mu_ranges[i, :] = test_mus[start], test_mus[end]

        return valid_mu_ranges

    # Positive values
    valid_mu_ranges_pos = get_mu_ranges(mu)
    valid_mu_range_pos = np.max(valid_mu_ranges_pos[:, 0]), np.min(valid_mu_ranges_pos[:, 1])
    if valid_mu_range_pos[1] - valid_mu_range_pos[0] > 0:
        output.append(valid_mu_range_pos)

    # Negative values
    valid_mu_ranges_neg= get_mu_ranges(-mu[::-1])
    valid_mu_range_neg = np.max(valid_mu_ranges_neg[:, 0]), np.min(valid_mu_ranges_neg[:, 1])
    if valid_mu_range_neg[1] - valid_mu_range_neg[0] > 0:
        output.append(valid_mu_range_neg)

    return output


def get_two_sided_uncertainty_from_loglkl(log_lkl, mu_hat, mu_ranges,
                                          verbose=True, N=1000):
    """Find sigmas on log_lkl surface."""
    # What about this:
    # 1- Get log_lkl(x) for x in logspace of each range
    # 2- If no minima found, skip frequency
    # 3- If only one minimum found, use that one
    # 5- Use results of that as seeds for scipy.minimize
    max_lkl = log_lkl(mu_hat)
    def f(mu):
        return (log_lkl(mu) - (max_lkl - .5))**2

    minima = []
    for j, (xmin, xmax) in enumerate(mu_ranges):
        sign = np.sign(xmin)
        x = sign*np.logspace(np.log10(sign*xmin), np.log10(sign*xmax), N)
        y = [f(xi) for xi in x]

        for i, val in enumerate(y):
            if i == 0 or i == len(y) - 1:
                continue
            if y[i-1] > val and y[i+1] > val:
                minima += [(x[i], j)]

    if not minima or len(minima) > 2:
        return None, None

    sigmas = np.zeros(len(minima))
    for i, (mu, j) in enumerate(minima):
        if mu < mu_hat:
            bound = [(mu_ranges[j][0], mu_hat)]
        else:
            bound = [(mu_hat, mu_ranges[j][1])]

        popt = minimize(f, mu, tol=1e-10,
                        method="Nelder-Mead", bounds=bound)
        assert popt.success
        sigmas[i] = popt.x

    # Convert positions to distances
    sigmas = np.sort(sigmas)
    if len(sigmas) == 1:
        if sigmas[0] < mu_hat:
            sigma_lo, sigma_hi = mu_hat - sigmas[0], None
        else:
            sigma_lo, sigma_hi = None, sigmas[0] - mu_hat
    else:
        sigma_lo, sigma_hi = mu_hat - sigmas[0], sigmas[1] - mu_hat

    if verbose and len(sigmas) == 2:
        x = np.linspace(mu_hat - sigma_lo*1.2, mu_hat + 1.2*sigma_hi, 1000)
        ax = plt.subplot(111)
        ax.set_yscale("log")
        ax.plot(x, np.array([f(xi) for xi in x]))
        ax.axvline(mu_hat, linestyle="--", color="r")
        ax.axvline(mu_hat-sigma_lo, linestyle="--", color="r")
        ax.axvline(mu_hat+sigma_hi, linestyle="--", color="r")
        ax2 = ax.twinx()
        ax2.plot(x, np.array([log_lkl(xi) for xi in x]), color="C1")
        plt.show()

    return sigma_lo, sigma_hi


def get_upper_limits_approx(Y, bkg, peak_norm, model_args, alpha_CL=.95):
    """Calculate an upper limit based on asymptotic formulae"""
    # Bkg model was already optimised, so take popt from model_args to get Lambda
    # Just need mu as free parameter
    def log_lkl(param):
        return log_likelihood(param, Y, bkg, peak_norm, model_args)

    # Get mu_hat / sigma from maximum likelihood
    # Start by checking mu validity range for each segment
    valid_mu_ranges = get_valid_mu_ranges(Y, bkg, peak_norm, model_args,
                                          start=-40, end=-30, n=1000)
    # Skip frequency entirely if no common range can be found for mu
    # TODO Instead, maybe just drop "worst" segment?
    if not valid_mu_ranges:
        return np.nan, np.nan

    # Now maximise lkl over all valid ranges and use best one
    max_lkls = np.zeros((len(valid_mu_ranges), 2))
    for i, mu_range in enumerate(valid_mu_ranges):
        # Get mu_guess from range and minimize
        mus = np.linspace(mu_range[0], mu_range[1], 100)
        lkl_values = np.array([log_lkl(xi) for xi in mus])
        popt = minimize(lambda x: -log_lkl(x), mus[np.argmax(lkl_values)],
                        method="Nelder-Mead", tol=1e-10,
                        bounds=[(mu_range[0], mu_range[1])])
        assert popt.success
        max_lkls[i, :] = -popt.fun, popt.x
    mu_hat = max_lkls[np.argmax(max_lkls[:, 0]), 1]

    # Calculate sigma using log lkl shape
    sigma_lo, sigma_hi = get_two_sided_uncertainty_from_loglkl(
        log_lkl, mu_hat, valid_mu_ranges, verbose=False)

    # What sigma do we use?
    sigma_lo = None if not sigma_lo else sigma_lo
    sigma_hi = None if not sigma_hi else sigma_hi
    if sigma_lo is None and sigma_hi is None:
        return np.nan, np.nan
    elif sigma_lo is None and sigma_hi is not None:
        sigma = sigma_hi
    elif sigma_lo is not None and sigma_hi is None:
        sigma = sigma_lo
    else:
        sigma = (sigma_lo + sigma_hi) / 2.

    # Careful, we calculate mu, and convert to Lambda later
    def opt_pval(param, _sigma):
        # Find the median of f_q_mu_0
        q_mu = np.linspace(0, max(25, (param/_sigma)**2+1), 1000)
        F_q_mu_0 = cumf_q_mu(q_mu, param, 0, _sigma)
        med_idx = np.where(F_q_mu_0 >= .5)[0][0]

        # Get p-value from f_q_mu_mu
        p_val = 1. - cumf_q_mu(np.ones(1)*q_mu[med_idx], param, param, _sigma)
        return (p_val - (1 - alpha_CL))**2

    # Calculate upper limit using cumf method
    try:
        popt_up = minimize(opt_pval, sigma, args=(sigma,),
                           bounds=[(0, None)], method="Nelder-Mead", tol=1e-10)
    except IndexError as err:
        print(mu_hat)
        print(valid_mu_ranges)
        print(sigma)
        get_two_sided_uncertainty_from_loglkl(
            log_lkl, mu_hat, valid_mu_ranges, verbose=True)
        raise err
    upper_lim, uncertainty = popt_up.x, sigma

    # Use previous results as seeds for full calculation
    # popt_profile = profile_calc(bkg, peak_norm, model_args, upper_lim, valid_mu_ranges[0],
    #                             alpha_CL=alpha_CL, n_q_mu=10000, no_tqdm=True)
    # if not isinstance(popt_profile, OptimizeResult)  or not popt_profile.success:
    #     return np.nan, np.nan
    # upper_lim, uncertainty = popt_profile.x, sigma

    # Convert back to Lambda_i^-1
    upper_Lambda = np.sqrt(upper_lim)
    sigma_Lambda = uncertainty / (2 * upper_Lambda)
    return upper_Lambda, sigma_Lambda


def process_segment(args):
    """Calculate an upper limit - wrapper for multiprocessing."""
    i, alpha_CL, freq_Hz, Y, bkg, model_args, peak_norm = args
    return i, freq_Hz, get_upper_limits_approx(Y, bkg, peak_norm,
                                               model_args, alpha_CL=alpha_CL)


def main():
    """Get all necessary data and launch analysis."""
    # Analysis variables
    n_frequencies = 2000
    num_processes = 11
    alpha_CL = 0.95
    _MAX_CHI_SQR = 10
    df_columns = ["x_knots", "y_knots", "alpha_skew", "loc_skew", "sigma_skew", "chi_sqr"]

    # TODO: rel. paths
    json_path = "data/processing_results.json"
    data_paths = ["data/result_epsilon10_1243393026_1243509654_H1.txt",
                  "data/result_epsilon10_1243393026_1243509654_L1.txt",
                  "data/result_epsilon10_1260438984_1260540984_L1.txt",
                  "data/result_epsilon10_1244106426_1244208890_H1.txt",
                  "data/result_epsilon10_1261705825_1261828599_H1.txt",
                  "data/result_epsilon10_1244106426_1244208890_L1.txt",
                  "data/result_epsilon10_1261705825_1261828599_L1.txt",
                  "data/result_epsilon10_1258629613_1258749121_H1.txt",
                  "data/result_epsilon10_1266964917_1267140541_H1.txt",
                  "data/result_epsilon10_1258629613_1258749121_L1.txt",
                  "data/result_epsilon10_1266964917_1267140541_L1.txt",
                  "data/result_epsilon10_1260438984_1260540984_H1.txt"]
    for data_path in glob.glob("data/*.lpsd"):
        data_paths.append(data_path)
    calib_path = "../shared_git_data/Calibration_factor_A_star.txt"

    # 0. Get A_star
    calib = np.loadtxt(calib_path, delimiter="\t")
    f_calib = {"H1": interp1d(calib[:, 1], calib[:, 0]),
               "L1": interp1d(calib[:, 1], calib[:, 0])}  # TODO: other file
    # 0.1 Open data HDFs
    data_info = []
    # ifos, data_hdfs = [], []
    for data_path in data_paths:
        _data_info = []
        _data_info.append(h5py.File(sensutils.get_corrected_path(data_path), "r"))
        _data_info.append(parse_ifo(data_path))
        df_key = "splines_" + sensutils.get_df_key(data_path)
        _data_info.append(sensutils.get_results(df_key, json_path))
        assert _data_info[-1] is not None
        data_info.append(_data_info)

    # ###### # ###### # ###### #
    test_frequencies = np.logspace(np.log10(calib[0, 1]), np.log10(calib[-1, 1]), n_frequencies)
    def args():
        count = 0
        for test_freq in test_frequencies:
            # Variables to "fill-in" for each data segment
            Y, bkg, model_args, peak_norm = [], [], [], []
            for hdf_path, ifo, df in data_info:
                # Find the df entry that contains the tested frequency
                mask = (df['fmin'] <= test_freq) & (df['fmax'] >= test_freq)
                # Skip this entry if test_freq is out of bounds
                if np.sum(np.array(mask, dtype=int)) == 0:
                    continue
                x_knots, y_knots, alpha_skew, loc_skew, sigma_skew, chi_sqr\
                    = df[mask][df_columns].iloc[0]
                # Skip this entry if fit is bad
                if chi_sqr >= _MAX_CHI_SQR:
                    continue

                # Find closest matching frequency
                frequencies = hdf_path["frequency"]
                frequency_idx = binary_search(frequencies, test_freq)
                # Take care of edge cases
                while frequencies[frequency_idx] < calib[0, 1]:
                    frequency_idx += 1
                while frequencies[frequency_idx] > calib[-1, 1]:
                    frequency_idx -= 1
                # frequency_idx = np.argmin(np.abs(frequencies - test_freq))
                freq_Hz = frequencies[frequency_idx]

                # Get calib factor
                A_star_sqr = f_calib[ifo](freq_Hz)**2

                # Fill argument lists
                Y.append(hdf_path["logPSD"][frequency_idx])
                bkg.append(models.model_xy_spline(
                    np.concatenate([x_knots, y_knots]), extrapolate=True)(
                        np.log(freq_Hz)))
                model_args.append([alpha_skew, loc_skew, sigma_skew])
                peak_norm.append(rho_local / (np.pi*freq_Hz**3 * A_star_sqr
                                              * (constants.e / constants.h)**2))
            if len(Y) == 0:
                continue
            Y, bkg, model_args, peak_norm = list(map(np.array, [Y, bkg, model_args, peak_norm]))
            yield (count, alpha_CL, freq_Hz,
                   *(np.array(x) for x in (Y, bkg, model_args, peak_norm)))
            count += 1  # Book-keeping for results merging

    # 2. Create job Pool
    results = []
    with Pool(num_processes, maxtasksperchild=10) as pool:
        with tqdm(total=n_frequencies, position=0, desc="Calc. upper lim") as pbar:
            for result in pool.imap_unordered(process_segment, args()):
                results.append(result)
                pbar.update(1)

    # 3. Merge results
    upper_limit_data = np.zeros((4, len(results)))
    for i, freq_Hz, (upper_limit, sigma) in results:
        upper_limit_data[0, i] = freq_Hz
        upper_limit_data[1, i] = upper_limit  # mu_upper, sigma
        upper_limit_data[2, i] = sigma

    # Clean NaNs
    mask = (np.isnan(upper_limit_data[1, :]) == 1) | (np.isnan(upper_limit_data[2, :]) == 1)
    upper_limit_data = upper_limit_data[:, ~mask]

    # 4. Plot!
    def smooth_curve(y, w):
        return sensutils.kde_smoothing(y, w)

    w = 33
    smooth_lim = np.exp(smooth_curve(np.log(upper_limit_data[1, :]), w))
    smooth_sigma = smooth_curve(upper_limit_data[2, :], w)
    geo_data = np.loadtxt("geo_limits.csv", delimiter=",")

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(geo_data[:, 0], geo_data[:, 1], label="GEO600", linewidth=4, color="C0")

    two = np.load("test.npy")
    _smooth_lim, smooth_sigma = np.exp(smooth_curve(np.log(two[1, :]), w)), smooth_curve(two[2, :], w)
    ax.plot(two[0, :], _smooth_lim, color="gold", label="Gold", linewidth=4.)
    f_gold = interp1d(two[0, :], _smooth_lim, bounds_error=False)

    ax.plot(upper_limit_data[0, :], smooth_lim,
            linewidth=4, color="silver", label="Gold+Silver")
    # ax.scatter(upper_limit_data[0, :], upper_limit_data[1, :],
    #            alpha=.33, color="C1", s=4)
    # ax.errorbar(upper_limit_data[0, :], upper_limit_data[1, :], upper_limit_data[2,:],
    #             color="C1", fmt=".")
    # ax.fill_between(upper_limit_data[0, :], smooth_lim+smooth_sigma, smooth_lim-smooth_sigma,
    #                 color="C1", alpha=.33)
    # ax.plot(upper_limit_data[0, :], smooth_lim / np.sqrt(2), label="L1->H1+L1 Projection",
    #         color="C3", linewidth=2.)

    np.save("full_plus1k.npy", upper_limit_data)
    # two = np.load("one_segment_H1_upper_lim.npy")
    # _smooth_lim, _ = np.exp(smooth_curve(np.log(two[1, :]), w)), smooth_curve(two[2, :], w)
    # ax.plot(two[0, :], _smooth_lim, color="C3", label="Initial estimate", linewidth=2.)

    x = upper_limit_data[0, :]
    y = smooth_lim/2.8*smooth_lim/(f_gold(x)/1.89)
    mask = x > 180
    ax.plot(x[mask], y[mask],
            linewidth=4, color="chocolate", label="Gold+Silver+Bronze (Projected)")
    # _smooth_lim, _ = np.exp(smooth_curve(np.log(two[1, :]), w)), smooth_curve(two[2, :], w)
    # ax.plot(two[0, :], _smooth_lim/1.89, color="C2", label="Silver projected", linewidth=2.)
    # ax.fill_between(two[0, :], smooth_lim+smooth_sigma, smooth_lim-smooth_sigma,
                    # color="C2", alpha=.33)

    # Nice things
    ax.set_yscale("log")
    ax.set_xscale("log")
    axTop = ax.twiny()
    axTop.set_xscale("log")
    ax.set_xlim(10, 5000)
    axTop.set_xlim(ax.get_xlim())
    ticks = np.array([1e-13, 1e-12, 1e-11])
    axTop.set_xticks(ticks * constants.e / constants.h)
    axTop.set_xticklabels(ticks)
    ax.set_xlabel("Frequency (Hz)")
    axTop.set_xlabel("DM mass (eV)")
    ax.set_ylabel(r"$1/\Lambda_i$ (GeV)")
    ax.set_title("PRELIMINARY 95% UPPER LIMITS")
    ax.legend(loc="best")
    ax.grid(color="grey", alpha=.33, linestyle="--", linewidth=1.5, which="both")
    ax.minorticks_on()
    plt.show()


if __name__ == '__main__':
    main()
