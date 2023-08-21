"""Calculate experimental upper limits based on real data."""
from multiprocessing import Pool
from tqdm import tqdm, trange
import numpy as np
import h5py
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import skewnorm
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize, minimize_scalar
from scipy.special import erf, erfc
from scipy.integrate import quad
from scipy import constants
from scipy.stats import norm
# Project imports
import sensutils
import models


# Global constant for peak normalisation in likelihood calculation
rho_local = 0.4 / (constants.hbar / constants.e * 1e-7 * constants.c)**3  # GeV/cm^3 to GeV^4
sqrt_two = np.sqrt(2)


def g_prime(f, alpha, mu, sigma):
    """First derivative of g with respect to f."""
    term1 = alpha * np.exp(-0.5 * ((f - mu)/sigma)**2 - 0.5 * alpha**2 * ((f - mu)/sigma)**2) / (np.pi * sigma**2)
    term2 = -(f - mu) * norm.sf(alpha * (f - mu) / (np.sqrt(2) * sigma)) / (np.exp((f - mu)**2 / (2 * sigma**2)) * np.sqrt(2 * np.pi) * sigma**3)
    return term1 + term2


def log_g_prime(f, alpha, mu, sigma):
    sum_term = alpha/np.pi*np.exp(-0.5*alpha**2*((f - mu)/sigma)**2) - (f - mu)/(np.sqrt(2*np.pi)*sigma)*norm.sf(alpha * (f - mu) / (np.sqrt(2) * sigma))
    log_common_term = -0.5*((f - mu)/sigma)**2 - 2*np.log(sigma)
    # if sum_term < 0:
    #     sum_term *= -1
    sign = -1 if sum_term < 0 else 1
    return log_common_term + np.log(sum_term * sign), sign


def g_double_prime(f, alpha, mu, sigma):
    """Second derivative of g with respect to f."""
    term1 = norm.sf(alpha * (f - mu) / (np.sqrt(2) * sigma)) / (np.sqrt(2*np.pi)*sigma**5) * np.exp(-0.5*((f - mu) / sigma)**2)
    term2 = (f - mu)**2 - sigma**2
    term3 = alpha*(f - mu)*(2 + alpha**2) / (np.pi * sigma**4) * np.exp(-0.5*((f - mu) / sigma)**2*(1 + alpha**2))
    return term1 * term2 - term3


def f_zero(mu, yi, betai, bkgi):
    """Just f."""
    return np.log(yi) - np.log(np.exp(bkgi) + betai*mu**2)


def f_prime(mu, betai, bkgi):
    """First derivative of f with respect to mu."""
    return -2*mu*betai / (np.exp(bkgi) + betai*mu**2)


def log_f_prime(mu, betai, bkgi):
    output = np.log(np.abs(-2*mu*betai)) - np.log((np.exp(bkgi) + betai*mu**2))
    sign = -1 if mu > 0 else 1
    return output, sign


def f_double_prime(mu, betai, bkgi):
    """Second derivative of f with respect to mu."""
    return -2*betai * (np.exp(bkgi) - betai*mu**2) / (np.exp(bkgi) + betai*mu**2)**2


def log_f_double_prime(mu, betai, bkgi):
    output = np.log(2*betai) + np.log(np.abs(-betai*mu**2 + np.exp(bkgi))) - 2*np.log(betai*mu**2 + np.exp(bkgi))
    sign = -1 if betai*mu**2 - np.exp(bkgi) < 0 else 1
    return output, sign


def get_Fisher(mu, yi, betai, bkgi, alpha, mu_skew, sigma):
    """Return the second derivative of the log-likelihood with respect to mu."""
    f = f_zero(mu, yi, betai, bkgi)
    g_p_p_f = g_double_prime(f, alpha, mu_skew, sigma)
    s_g_p_p_f = -1 if g_p_p_f < 0 else 1
    log_g_p_p_f  = np.log(g_p_p_f*s_g_p_p_f)

    log_g_p_f, s_g_p_f = log_g_prime(f, alpha, mu_skew, sigma)
    log_f_p, _ = log_f_prime(mu, betai, bkgi)
    log_f_p_p, s_f_p_p = log_f_double_prime(mu, betai, bkgi)

    term1 = np.exp(log_f_p_p + log_g_p_f) * s_f_p_p * s_g_p_f
    term2 = np.exp(2*log_f_p + log_g_p_p_f) * s_g_p_p_f
    return term1 + term2


def get_upper_limits_approx(X, Y_data, idx_peak, peak_norm, model_args,
                            alpha_CL=.95):
    """Calculate an upper limit based on asymptotic formulae"""
    # Bkg model was already optimised
    # So take popt from model_args to get Lambda
    # Just need mu_hat as free parameter
    x_knots, y_knots, alpha_skew, loc_skew, sigma_skew = model_args
    bkg = models.model_xy_spline(np.concatenate([x_knots, y_knots]), extrapolate=True)(X[idx_peak])

    def lkl(param):
        residuals = Y_data[idx_peak] - np.log(np.exp(bkg) + peak_norm*param**2)
        # return -sensutils.logpdf_skewnorm(residuals, loc_skew, sigma_skew, alpha_skew)
        return -skewnorm.pdf(residuals, alpha_skew, loc=loc_skew, scale=sigma_skew)

    mode_skew = sensutils.get_mode_skew(loc_skew, sigma_skew, alpha_skew)
    # mode_skew = 0  # TODO: correct?
    # TODO: in the current lkl form, mu_hat will always be positive
    sign = 1 if np.exp(Y_data[idx_peak] - mode_skew) - np.exp(bkg) > 0 else -1
    mu_hat = sign*np.sqrt(np.abs(np.exp(Y_data[idx_peak] - mode_skew) - np.exp(bkg)) / peak_norm)

    sigma_distr = sigma_skew*np.sqrt(1 - 2/np.pi*(alpha_skew**2 / (1 + alpha_skew**2)))
    sigma = np.abs(sigma_distr*np.exp(Y_data[idx_peak] - mode_skew) / (2. * peak_norm * mu_hat))

    x0 = mu_hat
    # x0 = np.sqrt(np.abs((np.exp(Y_data[idx_peak] - mode_skew) - np.exp(bkg)) / peak_norm))
    # popt = minimize_scalar(lkl, method="brent", bracket=(x0*0.99, x0*1.01),
    #                        options={"xtol": 1e-10, "maxiter": 100})
    popt = minimize(lkl, x0, method="Powell")

    # Now get sigma_mu_hat from d^2lnL/dmu^2
    Fisher = get_Fisher(popt.x, np.exp(Y_data[idx_peak]), peak_norm, bkg,
                        alpha_skew, loc_skew, sigma_skew)

    # TMP - Asymptotic likelihood answer

    print("mu_hats:", popt.x, mu_hat, "Mh")
    print("sigma:", sigma)
    print("beta:", peak_norm)
    print(popt)
    error
    # def get_q_mu_mu_prime(q_mu, mu_m, mu_prime):
    #     f_q_mu = 1 / (2*np.sqrt(2*np.pi*q_mu))*np.exp(
    #         -0.5*(np.sqrt(q_mu) - (mu_m - mu_prime)/sigma)**2)
    #     # f_q_mu[0] = norm.cdf((mu_prime - mu_m)/sigma)
    #     mask = q_mu > mu_m**2 / sigma**2
    #     f_q_mu[mask] = sigma / (2*np.sqrt(2*np.pi)*mu_m)*np.exp(
    #             -0.5*(q_mu[mask] - (mu_m**2 - 2*mu_m*mu_prime) / sigma**2)**2 / (mu_m / sigma)**2)

    #     # print(sigma, mu_m, mu_prime)
    #     # print(np.sqrt(q_mu[0]) - (mu_m - mu_prime)/sigma, f_q_mu[0])
    #     # print(-0.5*(q_mu[-1] - (mu_m**2 - 2*mu_m*mu_prime) / sigma**2)**2 / (mu_m / sigma)**2)
    #     # print("---")
    #     return f_q_mu

    # def pval(param):
    #     # Get med[f(q_mu|0)]
    #     q_mu = np.linspace(1e-5, 25, 1000)
    #     # TODO: make sure last value is small!
    #     f_q_mu_0 = get_q_mu_mu_prime(q_mu, param, 0)
    #     f_q_mu_mu = get_q_mu_mu_prime(q_mu, param, param)
    #     med_f_q_mu_0 = sensutils.get_distr_median(q_mu, f_q_mu_0)
    #     # print(med_f_q_mu_0)
    #     # print(np.sum(np.array(np.isnan(f_q_mu_0), dtype=int)))
    #     # print(np.sum(np.array(np.isnan(f_q_mu_mu), dtype=int)))
    #     # Use to calculate p-value on f(q_mu|mu)
    #     _norm, _ = quad(interp1d(q_mu, f_q_mu_mu), q_mu[0], q_mu[-1])
    #     p_val, _ = quad(interp1d(q_mu, f_q_mu_mu / _norm), med_f_q_mu_0, q_mu[-1])
    #     plt.figure()
    #     ax = plt.subplot(111)
    #     ax.set_yscale("log")
    #     ax.plot(q_mu, f_q_mu_0)
    #     ax.plot(q_mu, f_q_mu_mu)
    #     ax.axvline(med_f_q_mu_0, linestyle="--", color="r")
    #     ax.set_title(f"mu: {param[0]:.1e}, p: {p_val:.2f}")
    #     plt.show()
    #     return (p_val - (1 - alpha_CL))**2

    def get_q_mu_mu_prime(q_mu, mu_m, mu_prime):
        if q_mu > mu_m**2 / sigma**2:
            out = 1 / (2*np.sqrt(2*np.pi*q_mu))*np.exp(
            -0.5*(np.sqrt(q_mu) - (mu_m - mu_prime)/sigma)**2)
        else:
            out = 1 / (2*np.sqrt(2*np.pi*q_mu))*np.exp(
            -0.5*(np.sqrt(q_mu) - (mu_m - mu_prime)/sigma)**2)
        return out

    def get_pval(param):
        # Find the median of f_q_mu_0
        epsilon = 1e-5
        _norm, _ = quad(get_q_mu_mu_prime, epsilon, np.infty, args=(param, 0))
        def loss(_param):
            _int, _ = quad(get_q_mu_mu_prime, epsilon, _param, args=(param, 0))
            return (_int/_norm - .5)**2
        popt = minimize(loss, (param/sigma)**2, bounds=[(0, 100)],
                        method="Nelder-Mead")
        med_f_q_mu_0 = popt.x
        _int, _ = quad(get_q_mu_mu_prime, epsilon, med_f_q_mu_0, args=(param, 0))
        print("med:", med_f_q_mu_0, _int/_norm)

        # Calculate p-value of f_q_mu_mu
        _norm, _ = quad(get_q_mu_mu_prime, epsilon, 100, args=(param, param))
        pval, _ = quad(get_q_mu_mu_prime, med_f_q_mu_0, 100, args=(param, param))
        print("pval:", pval/_norm, pval, _norm)

        return (pval/_norm - (1 - alpha_CL))**2

    print("mu_hat, Sigma: ", popt.x, sigma)
    bounds = [(sigma*0.1, np.abs(popt.x)*5)]
    print("bounds", bounds)
    popt = minimize(get_pval, sigma, bounds=bounds,
                    method="Nelder-Mead", tol=1e-10)
                    # options={"gtol": 1e-20, "ftol": 1e-20})
    print(get_pval(popt.x), np.sqrt(popt.fun) + 0.05)
    print(popt)

    # Return mu_hat, sigma_dlnL, bkg
    return popt.x, np.sqrt(1. / np.abs(Fisher)), bkg


def process_segment(args):
    """Calculate an upper limit - wrapper for multiprocessing."""
    i, freq_subset, logPSD_subset, fi, lo, peak_norm, model_args = args
    return i, lo+fi, get_upper_limits_approx(freq_subset, logPSD_subset, fi, peak_norm, model_args)


def main():
    """Get all necessary data and launch analysis."""
    # Analysis variables
    n_frequencies = 50  # Test frequencies per segment
    num_processes = 1

    # TODO: rel. paths
    json_path = "data/processing_results.json"
    data_path = "data/result_epsilon10_1243393026_1243509654_H1.txt"
    calib_path = "../shared_git_data/Calibration_factor_A_star.txt"

    # 0. Get A_star
    calib = np.loadtxt(calib_path, delimiter="\t")
    f_calib = {"H1": interp1d(calib[:, 1], calib[:, 0]),
               "L1": interp1d(calib[:, 1], calib[:, 0])}  # TODO: other file
    # 0.1 Get data
    with h5py.File(sensutils.get_corrected_path(data_path)) as _f:
        peak_mask = np.array(_f["peak_mask"][()], dtype=bool)
        freq = np.log(_f["frequency"][()])
        logPSD = _f["logPSD"][()]
    # 0.2 Get pre-processing results
    df_key = "splines_" + sensutils.get_df_key(data_path)
    df = sensutils.get_results(df_key, json_path)
    assert df is not None

    # 1. Generate args
    ifo = data_path.split(".")[-2].split("_")[-1]
    jump_idx_space = np.where(~peak_mask)[0]
    del peak_mask
    def args():
        """Loop over segments and ifos to give data and args."""
        # Loop over segments/ifos
        count = 0
        for _, (frequencies, x_knots, y_knots, alpha_skew, loc_skew,
                sigma_skew, chi_sqr) in df.iterrows():
            [lo, hi] = jump_idx_space[frequencies]
            freq_subset = freq[lo:hi].copy()
            if np.exp(freq_subset[0]) < calib[0, 1] or np.exp(freq_subset[-1]) > calib[-1, 1]:
                continue
            logPSD_subset = logPSD[lo:hi].copy()
            test_frequencies = map(lambda x: int(x) - lo,
                                   np.linspace(lo, hi, n_frequencies+2)[1:-1])
            # Loop over frequencies
            # TODO: Faster to calculate for different freqs in one go in same segment?
            for fi in test_frequencies:
                # Calculate a normalisation constant to pass to the peak
                # This includes calibration and conversion
                freq_Hz = np.exp(freq_subset[fi])
                A_star_sqr = f_calib[ifo](freq_Hz)**2
                # peak_norm = np.log(rho_local / (np.pi*freq_Hz**3 * A_star_sqr
                #                                 * (constants.e / constants.h)**2))
                peak_norm = rho_local / (np.pi*freq_Hz**3 * A_star_sqr
                                         * (constants.e / constants.h)**2)
                # These are the arguments to pass to a single computation
                model_args = x_knots, y_knots, alpha_skew, loc_skew, sigma_skew
                yield (count, freq_subset, logPSD_subset, fi, lo, peak_norm, model_args)
                count += 1

    # 2. Create job Pool
    n_upper_limits = len(df) * n_frequencies
    with Pool(num_processes, maxtasksperchild=10) as pool:
        results = []
        with tqdm(total=n_upper_limits, position=0, desc="Calc. upper lim") as pbar:
            for result in pool.imap_unordered(process_segment, args()):
                results.append(result)
                pbar.update(1)

    # 3. Merge results
    upper_limit_data = np.zeros((5, len(results)))
    for i, fi, upper_limit in results:
        upper_limit_data[0, i] = np.exp(freq[fi])
        upper_limit_data[1:4, i] = upper_limit  # mu, sigma, bkg
        upper_limit_data[4, i] = logPSD[fi]
    idx = np.argsort(upper_limit_data[0, :])
    upper_limit_data = upper_limit_data[:, idx]

    # deleteme
    fig = plt.figure()
    gs = GridSpec(2, 1)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    print("M;HH", np.sum(np.array(upper_limit_data[1, :] < 0, dtype=int)))
    ax.plot(upper_limit_data[0, :], upper_limit_data[1, :])
    ax2.plot(upper_limit_data[0, :], upper_limit_data[2, :])

    # TMP - alternative calculation
    data = np.zeros((7, len(freq)))
    idx = 0
    for _, (frequencies, x_knots, y_knots, alpha_skew, loc_skew,
            sigma_skew, chi_sqr) in tqdm(df.iterrows(), total=len(df)):
        [lo, hi] = jump_idx_space[frequencies]
        if np.exp(freq[lo]) < calib[0, 1] or np.exp(freq[hi]) > calib[-1, 1]:
            continue

        _hi = idx + hi - lo
        # Save data
        data[0, idx:_hi] = freq[lo:hi]
        data[1, idx:_hi] = logPSD[lo:hi]
        freq_Hz = np.exp(freq[lo:hi])
        A_star_sqr = f_calib[ifo](freq_Hz)**2
        data[2, idx:_hi] = rho_local / (np.pi*freq_Hz**3 * A_star_sqr
                                        * (constants.e / constants.h)**2)
        # Save spline
        data[3, idx:_hi] = models.model_xy_spline(
            np.concatenate([x_knots, y_knots]), extrapolate=True)(logPSD[lo:hi])
        # Save skew params
        data[4, idx:_hi] = alpha_skew
        data[5, idx:_hi] = loc_skew
        data[6, idx:_hi] = sigma_skew

        # Loop condition
        idx += hi - lo

    # Sort data over frequency
    m = data[0, :] == 0
    idx = np.argsort(data[0, :][~m])
    data = data[:, ~m][:, idx]

    # Calculate upper limit
    mode_skew = sensutils.get_mode_skew(data[5, :], data[6, :], data[4, :])
    std_skew = data[6, :] * np.sqrt(1 - 2/np.pi*(data[4, :] / (1 + data[4, :]**2))**2)
    mu_hat = np.sqrt((np.exp(data[1, :] - mode_skew) - np.exp(data[3, :])) / data[2, :])
    mu_sigma = 0.5 * std_skew * np.exp(data[1, :] - mode_skew) / mu_hat

    # Convert results to "old" format
    pruning = 100
    upper_limit_data = np.zeros((3, data[0, ::pruning].shape[0]))
    upper_limit_data[0, :] = np.exp(data[0, ::pruning])
    upper_limit_data[1, :] = mu_hat[::pruning]
    upper_limit_data[2, :] = mu_sigma[::pruning]
    del data

    # deleteme
    ax.plot(upper_limit_data[0, :], upper_limit_data[1, :])
    ax2.plot(upper_limit_data[0, :], upper_limit_data[2, :])
    plt.show()
    return

    # 4. Plot!
    def smooth_curve(y, w):
        return sensutils.kde_smoothing(y, w)

    w = 10
    smooth_lim = smooth_curve(upper_limit_data[1, :] + 1.64*upper_limit_data[2, :],
                              w)
    smooth_sigma = smooth_curve(upper_limit_data[2, :],
                                w)
    plt.figure()
    ax = plt.subplot(111)
    ax.set_yscale("log")
    ax.set_xscale("log")


    ax.errorbar(upper_limit_data[0, :], smooth_lim, smooth_sigma,
                fmt=".", label="LIGO", linewidth=2., color="C1")
    geo_data = np.loadtxt("geo_limits.csv", delimiter=",")
    ax.plot(geo_data[:, 0], geo_data[:, 1], label="GEO600", linewidth=2, color="C0")

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
    # n = upper_limit_data.shape[1]
    # coupling = np.exp(-0.5*upper_limit_data[1, :n])
    # sigma = 0.5*coupling*upper_limit_data[2, :n]
    # ax.errorbar(upper_limit_data[0, :n], coupling+1.64*sigma, sigma, fmt=".")
    # ax.errorbar(upper_limit_data[0, :n], coupling, sigma, fmt=".")


if __name__ == '__main__':
    main()
