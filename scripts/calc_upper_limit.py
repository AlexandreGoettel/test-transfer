"""Calculate experimental upper limits based on real data."""
from multiprocessing import Pool
from tqdm import tqdm, trange
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import constants
from scipy.optimize import minimize
from scipy.stats import norm, skewnorm
from scipy.interpolate import interp1d
from scipy.special import erf
# Project imports
import sensutils
import models
import hist


# Global constant for peak normalisation in likelihood calculation
rho_local = 0.4 / (constants.hbar / constants.e * 1e-7 * constants.c)**3  # GeV/cm^3 to GeV^4


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


def profile_calc(Y, bkg, alpha_skew, loc_skew, sigma_skew,
                 alpha_CL=.95, n_q_mu=47500):
    """TMP - Test out full profile calculation for comparison."""
    mode_skew = sensutils.get_mode_skew(loc_skew, sigma_skew, alpha_skew)
    # def lkl(param):
    #     res = Y - np.log(np.exp(bkg) + param)
    #     return skewnorm.logpdf(res, alpha_skew, loc=loc_skew, scale=sigma_skew)
    # popt = minimize(lambda x: -lkl(x),
    #                 np.exp(Y - mode_skew) - np.exp(bkg),
    #                 method="Nelder-Mead")

    def lkl_new(param, data):
        res = data - np.log(np.exp(bkg) + param)
        return skewnorm.logpdf(res, alpha_skew, loc=loc_skew, scale=sigma_skew)

    # Generate a bunch of q_mus by fluctuating background
    def get_q_mu(data, mu_test):
        mu_hat = (np.exp(data - mode_skew) - np.exp(bkg))
        max_lkl = lkl_new(mu_hat, data)
        zero_lkl = lkl_new(0, data)

        q_mu = np.zeros_like(data)
        mask = mu_hat < mu_test
        q_mu[mask] = -2 * (lkl_new(mu_test, data[mask]) - max_lkl[mask])
        mask = mu_hat < 0
        q_mu[mask] = -2 * (lkl_new(mu_test, data[mask]) - zero_lkl[mask])
        return q_mu

    # n_q_mu_0, n_q_mu_mu = n_q_mu
    data_mu_0 = bkg + skewnorm.rvs(alpha_skew, loc=loc_skew, scale=sigma_skew, size=n_q_mu)
    full_bkg = bkg + skewnorm.rvs(alpha_skew, loc=loc_skew, scale=sigma_skew, size=n_q_mu)
    def get_p_val(mu_test, verbose=False):
        data_mu_mu = np.log(np.exp(full_bkg) + mu_test)
        q_mu_0 = get_q_mu(data_mu_0, mu_test)
        q_mu_mu = get_q_mu(data_mu_mu, mu_test)

        # Calculate p-value
        med_q_mu_0 = np.median(q_mu_0)
        k = np.sum(np.array(q_mu_mu >= med_q_mu_0, dtype=int))
        p_val, p_sigma = k / float(n_q_mu), np.sqrt(sensutils.get_eff_var(k, n_q_mu))
        if verbose:
            bins = np.linspace(0, max(q_mu_0), 100)
            ax = hist.plot_hist(q_mu_0, bins, logy=True, label="f(q_mu|0)", density=True)
            ax = hist.plot_hist(q_mu_mu, bins, ax=ax, label="f(q_mu|mu)", density=True)
            ax.axvline(med_q_mu_0, linestyle="--", color="r")
            ax.set_title(f"{p_val:.3f} +- {p_sigma:.4f}")

            x = np.linspace(bins[0], bins[-1], 1000)
            ax.plot(x, f_q_mu_tilda(x, mu_test, 0, sigma), label="f(q_mu|0)")
            ax.plot(x, f_q_mu_tilda(x, mu_test, mu_test, sigma), label="f(q_mu|mu)")
            ax.legend(loc="best")

            plt.show()
        return (p_val - (1 - alpha_CL))**2

    # Find good starting value
    sigma_distr = sigma_skew*np.sqrt(1 - 2/np.pi*(alpha_skew**2 / (1 + alpha_skew**2)))
    sigma = np.abs(sigma_distr*np.exp(Y - mode_skew))
    test = np.linspace(.1*sigma, 5*sigma, 50)
    diffs = [get_p_val(val) for val in test]
    # Minimize: get mu for which p-val is correct
    popt = minimize(get_p_val, test[np.argmin(diffs)],
                    bounds=[(0, 5*sigma)],
                    method="Nelder-Mead",
                    tol=1e-6)
    return popt


def get_upper_limits_approx(Y, bkg, peak_norm, model_args, alpha_CL=.95):
    """Calculate an upper limit based on asymptotic formulae"""
    # Bkg model was already optimised, so take popt from model_args to get Lambda
    # Just need mu as free parameter
    def log_lkl(param):
        residuals = Y - np.log(np.exp(bkg) - peak_norm*param)
        lkl = [sensutils.logpdf_skewnorm(res, alpha, loc, omega)
               for res, [alpha, loc, omega] in zip(residuals, model_args)]
        return np.sum(lkl)

    # Get mu_hat / sigma from maximum likelihood
    # Starting param from first data point
    # Remember: mu = Lambda_i^-2, and mu_hat can be negative (but not mu_upper)
    mode_skew = sensutils.get_mode_skew(*model_args[0, :])
    initial_guess = (np.exp(Y[0] - mode_skew) - np.exp(bkg[0])) / peak_norm[0]
    # Nan-protection for initial guess
    if np.isnan(log_lkl(initial_guess)) or np.isinf(log_lkl(initial_guess)):
        x = np.sign(initial_guess)*np.logspace(-40, -30, 100)
        y = np.array([log_lkl(xi) for xi in x])
        mask = (np.isnan(y) == 1) ^ (np.isinf(y) == 1)
        if not np.sum(~mask):
            x = -x
            y = np.array([log_lkl(xi) for xi in x])
            mask = (np.isnan(y) == 1) ^ (np.isinf(y) == 1)
            tqdm.write("FAILED ITER")
            if not np.sum(~mask):
                return np.nan, np.nan
        # initial_guess = x[~mask][np.argmin(x[~mask] - initial_guess)]
        initial_guess = x[~mask][np.argmax(y[~mask])]
    popt = minimize(lambda x: -log_lkl(x), initial_guess,
                    method="Nelder-Mead", tol=1e-10)
    assert popt.success

    if popt.fun > 10:
        return np.nan, np.nan

    # Calculate sigma using log lkl shape
    max_lkl = log_lkl(popt.x)
    # Estimate starting param for sigma
    broad_initial_guess_sigma = np.std((np.exp(skewnorm.rvs(*model_args[0, :], size=1000) + bkg[0]
                                               - mode_skew) - np.exp(bkg[0]))/peak_norm[0])
    x = np.linspace(0, 5*broad_initial_guess_sigma, 100)
    y = [(log_lkl(popt.x + xi) - (max_lkl - .5))**2 for xi in x]
    initial_guess = x[np.argmin(y)]
    # popt_lo = minimize(lambda x: (log_lkl(x) - (max_lkl - .5))**2, popt.x - .1*np.abs(popt.x),
    #                    method="Nelder-Mead", bounds=[(None, popt.x)])
    # sigma_lo = popt.x - popt_lo.x
    # tqdm.write("Up")
    popt_hi = minimize(lambda x: (log_lkl(popt.x + x) - (max_lkl - .5))**2, initial_guess,
                       method="Nelder-Mead", bounds=[(0, None)])
    # tqdm.write("Lo")
    sigma_hi = popt_hi.x[0]
    if sigma_hi == 0:
        print("mhhhhhh")
        print(popt)
        print(popt.x, broad_initial_guess_sigma)
        print(initial_guess)
        print(log_lkl(popt.x+initial_guess))
        print(popt_hi)
        plt.plot(x, y)
        print(y)
        plt.show()
        error
    # plt.plot(x, y)
    # plt.show()
    # error

    # Calculate using cumf and full and compare
    # Careful, we calculate on mu, and convert to Lambda later
    def opt_pval(param, _sigma):
        # Find the median of f_q_mu_0
        q_mu = np.linspace(0, max(25, (param/_sigma)**2+1), 1000)
        F_q_mu_0 = cumf_q_mu(q_mu, param, 0, _sigma)
        med_idx = np.where(F_q_mu_0 >= .5)[0][0]

        # Get p-value from f_q_mu_mu
        p_val = 1. - cumf_q_mu(np.ones(1)*q_mu[med_idx], param, param, _sigma)
        return (p_val - (1 - alpha_CL))**2

    # popt = profile_calc(Y, bkg, alpha_skew, loc_skew, sigma_skew, alpha_CL)
    popt_up = minimize(opt_pval, sigma_hi, args=(sigma_hi,),
                       bounds=[(0, None)], method="Nelder-Mead", tol=1e-10)
    if np.sqrt(popt_up.x) < 1e-20:
        print(popt)
        print(popt_hi)
        print(popt_up)
        error
    upper_lim, uncertainty = popt_up.x, sigma_hi
    # popt_lo = minimize(opt_pval, sigma_lo, args=(sigma_lo,),
    #                    bounds=[(0, None)], method="Nelder-Mead", tol=1e-10)
    # print(mu_hat, sigma_lo, sigma_hi)
    # print(popt_lo)
    # print("----")
    # print(popt_hi)

    # # Calculate using full profile calculation
    # test_mu = []
    # for _ in trange(10):
    #     Y = bkg + skewnorm.rvs(alpha_skew, loc=loc_skew, scale=sigma_skew)
    #     popt = profile_calc(Y, bkg, alpha_skew, loc_skew, sigma_skew, alpha_CL)
    #     test_mu.append(popt.x)

    # print("---")
    # print(sigma_lo, sigma_hi)
    # print(popt)
    # print(np.mean(test_mu), np.std(test_mu))
    # error

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
    num_processes = 1
    alpha_CL = 0.95
    _MAX_CHI_SQR = 10
    df_columns = ["x_knots", "y_knots", "alpha_skew", "loc_skew", "sigma_skew", "chi_sqr"]

    # TODO: rel. paths
    json_path = "data/processing_results.json"
    data_paths = ["data/result_epsilon10_1243393026_1243509654_L1.txt",
                  "data/result_epsilon10_1243393026_1243509654_H1.txt"]
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
    test_frequencies = np.linspace(calib[0, 1], calib[-1, 1], n_frequencies)
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
            yield (count, alpha_CL, freq_Hz,
                   *(np.array(x) for x in (Y, bkg, model_args, peak_norm)))
            count += 1  # Book-keeping for results merging

    # 2. Create job Pool
    with Pool(num_processes, maxtasksperchild=10) as pool:
        results = []
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
    mask = (np.isnan(upper_limit_data[1, :]) == 1) ^ (np.isnan(upper_limit_data[2, :]) == 1)
    upper_limit_data = upper_limit_data[:, ~mask]

    # 4. Plot!
    def smooth_curve(y, w):
        return sensutils.kde_smoothing(y, w)

    w = 33
    smooth_lim = smooth_curve(upper_limit_data[1, :], w)
    smooth_sigma = smooth_curve(upper_limit_data[2, :], w)
    geo_data = np.loadtxt("geo_limits.csv", delimiter=",")

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(upper_limit_data[0, :], smooth_lim,
            linewidth=2, color="C1", label="LIGO")
    ax.scatter(upper_limit_data[0, :], upper_limit_data[1, :])
    ax.fill_between(upper_limit_data[0, :], smooth_lim+smooth_sigma, smooth_lim-smooth_sigma,
                    color="C1", alpha=.33)
    ax.plot(geo_data[:, 0], geo_data[:, 1], label="GEO600", linewidth=2, color="C0")

    # Deleteme
    np.save("sigma_hi.npy", upper_limit_data)

    # Nice things
    ax.set_yscale("log")
    ax.set_xscale("log")
    axTop = ax.twiny()
    axTop.set_xscale("log")
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
