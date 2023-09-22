"""Calculate experimental upper limits based on real data."""
import os
from multiprocessing import Pool
from tqdm import tqdm, trange
import glob
import h5py
from findiff import FinDiff
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import constants
from scipy.optimize import minimize, newton, OptimizeResult
from scipy.stats import norm, skewnorm, chi2
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


def log_likelihood(params, Y, bkg, peak_norm, model_args,
                   calib_args=None, doCalib=False):
    """Likelihood of finding dark matter in the data!"""
    # Actual likelihood calculation
    if doCalib:
        mu_DM, theta = params
        residuals = 2*np.log(theta) + Y - np.log(np.exp(bkg) + peak_norm*mu_DM)
        gaussian_pull_term = sensutils.logpdf_skewnorm(  # skew-norm with zero skew is normal!
            theta, 0, loc=calib_args[:, 0], scale=calib_args[:, 1])
    else:
        mu_DM = params
        residuals = Y - np.log(np.exp(bkg) + peak_norm*mu_DM)
        gaussian_pull_term = 0

    skewnorm_term = sensutils.logpdf_skewnorm(
        residuals, model_args[:, 0], model_args[:, 1], model_args[:, 2])
    lkl = skewnorm_term + gaussian_pull_term
    # Security
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
    out[mask] = 0.5 / (np.sqrt(2*np.pi)*mu/sigma) * np.exp(-0.5*(
        (q_mu[mask] - (mu**2 - 2*mu*mu_prime)/sigma**2) / (2*mu/sigma))**2)
    return out


def cumf_q_mu(q_mu, mu, mu_prime, sigma):
    """Cumulative probability function for f(q_mu|mu_prime)."""
    out = norm.cdf(np.sqrt(q_mu) - (mu - mu_prime) / sigma)
    mask = q_mu > (mu / sigma)**2
    if len(mask):
        out[mask] = norm.cdf((q_mu[mask] - (mu**2 - 2*mu*mu_prime) / sigma**2) / (2*mu/sigma))
    return out


# TODO re-implement with extended (calib) likelihood
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
    """Find the indices of the leading and trailing groups of invalid numbers in an array."""
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


def get_two_sided_uncertainty_from_loglkl(log_lkl, mu_hat, sigma_est, mu_ranges,
                                          theta_hat=None, do_calib=False, N=1000):
    """Find sigmas on log_lkl surface."""
    # Get correct mu_range
    if mu_hat > 0:
        if mu_ranges[0][0] > 0:
            mu_range = mu_ranges[0]
        else:
            mu_range = mu_ranges[1]
    else:
        if mu_ranges[0][0] < 0:
            mu_range = mu_ranges[0]
        else:
            mu_range = mu_ranges[1]
    # Set search bounds
    delta_lo, delta_hi = sigma_est, sigma_est
    if mu_hat - delta_lo < mu_range[0]:
        delta_lo = (mu_hat - mu_range[0]) / 2.
    if mu_hat + delta_hi > mu_range[1]:
        delta_hi = (mu_range[1] - mu_hat) / 2.
    assert delta_lo > 0 and delta_hi > 0

    if do_calib:
        max_lkl = log_lkl([mu_hat, theta_hat])
        # Get the profile likelihood distance function
        x = np.linspace(mu_hat - 2*delta_lo, mu_hat + 2*delta_hi, N)
        profile_lkl = np.zeros_like(x)
        for i, xi in enumerate(x):
            popt = minimize(lambda theta: -log_lkl([xi, theta]), theta_hat,
                            method="Nelder-Mead", tol=1e-10, bounds=[(0.75, 1.25)])
            profile_lkl[i] = -2*(-popt.fun - max_lkl)
        # Check if the range is ok on lower side (could use protection for upper side?)
        if np.where(~np.isinf(profile_lkl))[0][0]:
            delta_lo = (mu_hat - x[np.where(~np.isinf(profile_lkl))[0][0]]) / 2.
            assert delta_lo > 0
            x = np.linspace(mu_hat - 2*delta_lo, mu_hat + 2*delta_hi, N)
            profile_lkl = np.array([-2*(log_lkl([xi, theta_hat]) - max_lkl) for xi in x])

        y = (profile_lkl - min(profile_lkl[~np.isnan(profile_lkl)]) - chi2.ppf(0.68, df=1))**2
        f = interp1d(x, y, bounds_error=True)

    else:
        max_lkl = log_lkl([mu_hat, theta_hat])
        # Get the profile likelihood distance function
        x = np.linspace(mu_hat - 2*delta_lo, mu_hat + 2*delta_hi, N)
        profile_lkl = np.array([-2*(log_lkl([xi, theta_hat]) - max_lkl) for xi in x])
        # Check if the range is ok on lower side (could use protection for upper side?)
        if np.where(~np.isinf(profile_lkl))[0][0]:
            delta_lo = (mu_hat - x[np.where(~np.isinf(profile_lkl))[0][0]]) / 2.
            assert delta_lo > 0
            x = np.linspace(mu_hat - 2*delta_lo, mu_hat + 2*delta_hi, N)
            profile_lkl = np.array([-2*(log_lkl([xi, theta_hat]) - max_lkl) for xi in x])

        mask = ~np.isnan(profile_lkl)
        _min = min(profile_lkl[mask])
        def f(xi):
            return np.log((-2*(log_lkl([xi, theta_hat]) - max_lkl) - _min - chi2.ppf(0.68, df=1))**2)

    # Find initial estimates on f(x)
    _x = x[x > mu_hat]
    _y = np.array([f(xi) for xi in _x])
    mask = np.isnan(_y) | np.isinf(_y)
    delta_hi = _x[~mask][np.argmin(_y[~mask])] - mu_hat
    # Delta_lo now
    _x = x[x < mu_hat]
    _y = np.array([f(xi) for xi in _x])
    mask = np.isnan(_y) | np.isinf(_y)
    delta_lo = mu_hat - _x[~mask][np.argmin(_y[~mask])]

    # Get min pos on both sides from interpolated function
    popt_hi = minimize(f, mu_hat + delta_hi,
                       method="Nelder-Mead", tol=1e-10, bounds=[(mu_hat, None)])
    popt_lo = minimize(f, mu_hat - delta_lo,
                       method="Nelder-Mead", tol=1e-10, bounds=[(None, mu_hat)])

    sigma_lo = abs(popt_lo.x - mu_hat) if popt_lo.fun < -5 else None
    sigma_hi = abs(popt_hi.x - mu_hat) if popt_hi.fun < -5 else None
    return sigma_lo, sigma_hi


def get_upper_limits_approx(Y, bkg, peak_norm, model_args, calib_args,
                            do_calib=False, use_sigma="Fisher", alpha_CL=.95, tol=1e-10):
    """Calculate an upper limit based on asymptotic formulae"""
    # Bkg model was already optimised, so take popt from model_args to get Lambda
    if do_calib:
        def log_lkl(params):
            return log_likelihood(params, Y, bkg, peak_norm, model_args,
                                calib_args, doCalib=True)
    else:
        def log_lkl(params):
            return log_likelihood(params[0], Y, bkg, peak_norm, model_args)

    # Get mu_hat / sigma from maximum likelihood
    # Start by checking mu validity range for each segment
    valid_mu_ranges = get_valid_mu_ranges(Y, bkg, peak_norm, model_args,
                                          start=-43, end=-33, n=1000)
    # Skip frequency entirely if no common range can be found for mu
    if not valid_mu_ranges:
        return np.nan, np.nan

    # Now maximise lkl over all valid ranges and use best one
    max_lkls = np.zeros((len(valid_mu_ranges), 3))
    for i, mu_range in enumerate(valid_mu_ranges):
        # Get initial guess from valid range
        mus = np.sign(mu_range[0])*np.logspace(np.log10(abs(min(mu_range))),
                                               np.log10(abs(max(mu_range))),
                                               15)
        max_lkl, max_x = -np.inf, [np.nan, np.nan]
        for mu in mus:
            if do_calib:
                popt = minimize(lambda x: -log_lkl(x),
                                [mu, 1],  # starting guess for eta_R is 1
                                method="Nelder-Mead",
                                tol=tol,
                                bounds=[(min(mu_range), max(mu_range)), (.75, 1.33)])
            else:
                popt = minimize(lambda x: -log_lkl(x),
                                mu,
                                method="Nelder-Mead",
                                tol=tol,
                                bounds=[(min(mu_range), max(mu_range))])
            if not popt.success:
                continue
            if -popt.fun != 0 and not np.isnan(-popt.fun) and not np.isinf(-popt.fun)\
                    and -popt.fun > max_lkl:
                max_lkl, max_x = -popt.fun, popt.x

        # Book keeping
        max_lkls[i, :len(max_x)+1] = max_lkl, *max_x
    mu_hat = max_lkls[np.argmax(max_lkls[:, 0]), 1]
    theta_hat = max_lkls[np.argmax(max_lkls[:, 0]), 2]  # this is zero if do_calib is False

    # Calculate sigma using log lkl shape
    # Base initial dx estimate on distance to edge of validity
    if mu_hat > 0:
        max_dist = min(mu_hat - valid_mu_ranges[0][0], valid_mu_ranges[0][1] - mu_hat)
    else:
        max_dist = min(mu_hat - valid_mu_ranges[1][0], valid_mu_ranges[1][1] - mu_hat)
    # Make sure that mu_hat is in an actual maximum
    if log_lkl([mu_hat - max_dist, theta_hat]) >= max(max_lkls[:, 0]) or\
        log_lkl([mu_hat + max_dist, theta_hat]) >= max(max_lkls[:, 0]) or\
            np.isinf(log_lkl([mu_hat - abs(mu_hat)*0.0001, theta_hat])):
        return np.nan, np.nan
    try:
        sigma = sensutils.sigma_at_point(lambda x: log_lkl([x, theta_hat]),
                                         mu_hat,
                                         initial_dx=min(max_dist/2., abs(mu_hat)),
                                         tolerance=1e-4)
    except ValueError:
        return np.nan, np.nan

    if use_sigma != "Fisher":
        sigma_lo, sigma_hi = get_two_sided_uncertainty_from_loglkl(
            log_lkl, mu_hat, sigma, valid_mu_ranges,
            do_calib=do_calib, theta_hat=theta_hat, N=1000)

    if use_sigma == "mean":
        if None in [sigma_lo, sigma_hi]:
            sigma = sigma_lo if sigma_hi is None else sigma_hi
        else:
            sigma = np.mean([sigma_lo, sigma_hi])

    elif use_sigma == "lower":  # Use the sigma on the zero side, relevant for upper lim!
        if None in [sigma_lo, sigma_hi]:
            sigma = sigma_lo if sigma_hi is None else sigma_hi
        elif mu_hat < 0:
            sigma = sigma_hi
        else:
            sigma = sigma_lo

    elif use_sigma != "Fisher":
        raise ValueError(f"Unknown value for 'use_sigma': '{use_sigma}'...")

    if sigma is None:
        return np.nan, np.nan

    # Careful, we calculate the limit on mu, and convert to Lambda later
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
                        bounds=[(0, None)], method="Nelder-Mead", tol=tol)
    except IndexError as e:
        tqdm.write(f"{sigma:.1e}")
        raise e
    upper_lim, uncertainty = popt_up.x, sigma

    # Convert back to Lambda_i^-1
    upper_Lambda = np.sqrt(upper_lim)
    sigma_Lambda = uncertainty / (2 * upper_Lambda)
    return upper_Lambda, sigma_Lambda


def process_segment(args):
    """Calculate an upper limit - wrapper for multiprocessing."""
    i, alpha_CL, freq_Hz, do_calib, use_sigma, Y, bkg, model_args, peak_norm, calib_args = args
    kwargs = {"do_calib": do_calib, "use_sigma": use_sigma, "alpha_CL": alpha_CL}
    return i, freq_Hz, get_upper_limits_approx(Y, bkg, peak_norm, model_args,
                                               calib_args, **kwargs)


def main():
    """Get all necessary data and launch analysis."""
    # Analysis variables
    n_frequencies = 2000
    num_processes = 11
    alpha_CL = 0.95
    _MAX_CHI_SQR = 10
    do_calib = False
    use_sigma = "lower"
    df_columns = ["x_knots", "y_knots", "alpha_skew", "loc_skew", "sigma_skew", "chi_sqr"]

    # TODO: rel. paths
    json_path = "data/processing_results.json"
    data_paths = []
    for data_path in glob.glob("data/result_epsilon10*"):
        data_paths.append(data_path)
    # data_paths = ["data/result_epsilon10_1239824770_1239926064_L1.lpsd",
    #               "data/result_epsilon10_1240305825_1240473003_L1.lpsd"]
    transfer_function_path = "../shared_git_data/Calibration_factor_A_star.txt"
    calib_dir = "data/calibration"

    # 0. Get A_star
    tf = np.loadtxt(transfer_function_path, delimiter="\t")
    f_A_star = {"H1": interp1d(tf[:, 1], tf[:, 0]),
                "L1": interp1d(tf[:, 1], tf[:, 0])}  # TODO: other file
    # 0.1 Open O3a and O3b calibration envelope files
    calib_gps_times = {
        "H1": [int(f.split("_")[4]) for f in glob.glob(os.path.join(
            calib_dir, "H1", "*FinalResults.txt"))],
        "L1": [int(f.split("_")[4]) for f in glob.glob(os.path.join(
            calib_dir, "L1", "*FinalResults.txt"))]
    }
    # 0.2 Open data HDFs
    data_info = []
    for data_path in tqdm(data_paths, desc="Opening HDFs", leave=False):
        _data_info = []
        _data_info.append(h5py.File(sensutils.get_corrected_path(data_path), "r"))
        ifo = parse_ifo(data_path)
        _data_info.append(ifo)
        df_key = "splines_" + sensutils.get_df_key(data_path)
        _data_info.append(sensutils.get_results(df_key, json_path))
        assert _data_info[-1] is not None
        # Calibration - Get closest calib GPS time and read out
        data_gps_time = int(os.path.split(data_path)[-1].split("_")[-3])
        nearest_calib_time = calib_gps_times[ifo][np.argmin(np.abs(
            np.array(calib_gps_times[ifo]) - data_gps_time))]
        calib_file = glob.glob(os.path.join(calib_dir, ifo,
                                            f"*{nearest_calib_time}*FinalResults.txt"))[0]
        data = np.loadtxt(calib_file)
        # Assume gaussian error
        freq, mag, mag_1sigma = data[:, 0], data[:, 1], data[:, 5]
        # print(np.max(mag+mag_1sigma), data[np.argmax(mag+mag_1sigma), 0])
        # assert np.max(mag+mag_1sigma) < 1.33
        f_calib_mag = interp1d(freq, mag, bounds_error=False)
        f_calib_sigma = interp1d(freq, np.abs(mag_1sigma - mag), bounds_error=False)
        # "Return"
        _data_info.append((f_calib_mag, f_calib_sigma))
        data_info.append(_data_info)

    # ###### # ###### # ###### #
    test_frequencies = np.logspace(np.log10(tf[0, 1]), np.log10(tf[-1, 1]), n_frequencies)
    def args():
        count = 0
        for test_freq in test_frequencies:
            # Variables to "fill-in" for each data segment
            Y, bkg, model_args, peak_norm, calib_args = [], [], [], [], []
            for hdf_path, ifo, df, (f_calib_mag, f_calib_sigma) in data_info:
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
                while frequencies[frequency_idx] < tf[0, 1]:
                    frequency_idx += 1
                while frequencies[frequency_idx] > tf[-1, 1]:
                    frequency_idx -= 1
                # frequency_idx = np.argmin(np.abs(frequencies - test_freq))
                freq_Hz = frequencies[frequency_idx]

                # Get tf factor
                A_star_sqr = f_A_star[ifo](freq_Hz)**2

                # Get calibration mu and sigma
                calib_mu, calib_sigma = f_calib_mag(freq_Hz), f_calib_sigma(freq_Hz)

                # Fill argument lists
                Y.append(hdf_path["logPSD"][frequency_idx])
                bkg.append(models.model_xy_spline(
                    np.concatenate([x_knots, y_knots]), extrapolate=True)(
                        np.log(freq_Hz)))
                model_args.append([alpha_skew, loc_skew, sigma_skew])
                peak_norm.append(rho_local / (np.pi*freq_Hz**3 * A_star_sqr
                                              * (constants.e / constants.h)**2))
                calib_args.append([calib_mu, calib_sigma])

            if len(Y) == 0:
                continue
            yield (count, alpha_CL, freq_Hz, do_calib, use_sigma,
                   *(np.array(x) for x in (Y, bkg, model_args, peak_norm, calib_args)))
            count += 1  # Book-keeping for results merging later

    # 2. Create job Pool
    results = []
    with Pool(num_processes, maxtasksperchild=10) as pool:
        with tqdm(total=n_frequencies, position=0, desc="Calc. upper lim") as pbar:
            for result in pool.imap_unordered(process_segment, args()):
                results.append(result)
                pbar.update(1)
    # for arg in tqdm(args(), total=n_frequencies):
    #     results.append(process_segment(arg))

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

    ax.scatter(upper_limit_data[0, :], upper_limit_data[1, :],
               alpha=.33, color="C1", s=4)
    # ax.errorbar(upper_limit_data[0, :], upper_limit_data[1, :], upper_limit_data[2,:],
    #             color="C1", fmt=".")
    ax.fill_between(upper_limit_data[0, :], smooth_lim+smooth_sigma, smooth_lim-smooth_sigma,
                    color="C1", alpha=.33)
    ax.plot(upper_limit_data[0, :], smooth_lim , label="LIGO",
            color="C1", linewidth=2.)

    np.save("silver_calib_calib.npy", upper_limit_data)
    two = np.load("silver_nocalib.npy")
    _smooth_lim, _ = np.exp(smooth_curve(np.log(two[1, :]), w)), smooth_curve(two[2, :], w)
    ax.plot(two[0, :], _smooth_lim, color="C3", label="nocalib", linewidth=2.)

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
