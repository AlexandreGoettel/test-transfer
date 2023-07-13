"""Calculate experimental upper limits based on simulated data with two ifos."""
import time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import skewnorm
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize
from scipy.special import erf, erfc
from scipy import constants
# Project imports
import utils
import models


# Global constant for peak normalisation in likelihood calculation
rho_local = 0.4 / (constants.hbar / constants.e * 1e-7 * constants.c)**3  # GeV/cm^3 to GeV^4
# epsilon = 1e-6
epsilon = 1  # FIXME
gamma = rho_local * constants.h**2 / (np.pi*epsilon*constants.e**2)


def logpdf_skewnorm(x, loc, scale, alpha):
    """Implement skewnorm.logpdf for speed."""
    norm_term = -0.5*((x - loc) / scale)**2
    skew_term = np.log(0.5*(1. + erf(alpha*(x - loc) / (np.sqrt(2)*scale))))
    return np.log(2) - 0.5*np.log(2*np.pi) + norm_term + skew_term

def get_mode_skew(mu, sigma, alpha):
    """Return (wikipedia) numerical approximation of mode."""
    delta = alpha / np.sqrt(1 + alpha**2)
    mu_z = np.sqrt(2 / np.pi) * delta
    sigma_z = np.sqrt(1 - mu_z**2)
    gamma_1 = (4 - np.pi) / 2. * mu_z**3 / (1 - mu_z**2)**1.5
    m0 = mu_z - gamma_1*sigma_z/2. - np.sign(alpha)/2.*np.exp(-2*np.pi/np.abs(alpha))
    return mu + sigma*m0


def second_derivative(alpha, sigma, Lambda, theta):
    """Second derivative of the log skew normal of (y-cubic(x)-theta)/sigma w.r.t. theta."""
    term1 = -2 * alpha**2 / (np.exp(alpha**2 * (Lambda - theta)**2 / sigma**2) * np.pi * sigma**2\
        * erfc(-alpha * (Lambda - theta) / (np.sqrt(2) * sigma))**2)
    term2 = -alpha**3 * np.sqrt(2 / np.pi)\
        * (Lambda - theta) / (np.exp(alpha**2 * (Lambda - theta)**2 / (2 * sigma**2)) * sigma**3\
        * erfc(-alpha * (Lambda - theta) / (np.sqrt(2) * sigma)))
    return term1 + term2 - sigma**-2


def bkg_model(cube, x_data, x_knots, shift=3):
    """Return log-log spline background."""
    y_knots = cube[shift:len(x_knots)+shift]
    return CubicSpline(x_knots, y_knots, extrapolate=False)(x_data)


def peak_model(mu, log_beta):
    """
    Normalised peak (currently delta) * coupling to PSD conversion.
    
    # Because of the small size of the peak, only return non-zero bins
    # starting at idx_peak (correct when not using delta peak?)
    mu = -2*ln(Lambda_i^-1) (so I can keep the linear dependence on mu)
    """
    return np.ones(1) * (log_beta - mu)


def likelihood(cube, idx_peak, freq, logPSD_data,
               x_knots, peak_norm, shift=3):
    """
    Calculate log Skew normal of residuals to model().
    
    Sum over both interferometers.
    cube: mu, theta_H1, theta_H2
    """
    # Separate parameters
    mu = cube[0]
    theta = np.reshape(cube[1:], (2, len(cube[1:])//2))

    # Sum over detectors to calculate likelihood
    total_likelihood = 0.
    for i, logPSD in enumerate(logPSD_data):
        y_model = bkg_model(np.concatenate([[mu], theta[i, :]]),
                            freq, x_knots, shift=shift)
        y_peak = peak_model(mu, peak_norm[i])
        y_model[idx_peak:idx_peak+len(y_peak)] = np.log(
            np.exp(y_peak) + np.exp(y_model[idx_peak:idx_peak+len(y_peak)]))

        alpha_skew, sigma_skew = theta[i, 0], np.sqrt(np.abs(theta[i, 1]))
        lkl = logpdf_skewnorm(logPSD - y_model,
                              -get_mode_skew(0, alpha_skew, sigma_skew),
                              sigma_skew,
                              alpha_skew)

        # Protect against inf
        # lkl_inf = np.isinf(lkl)
        # if np.sum(np.array(lkl_inf, dtype=int)):
        #     # This will give theoretically incorrect values
        #     # But only in cases to-be-rejected anyways
        #     try:
        #         lkl[lkl_inf] = np.min(lkl[~lkl_inf])
        #     except ValueError:
        #         lkl = -np.ones_like(lkl)*1e5
        total_likelihood += np.sum(lkl)

    return total_likelihood


def get_upper_limits_approx(X, Y_data, idx_peak, peak_norm):
    """Calculate an upper limit based on asymptotic formulae"""
    # TODO: pass pre-optimized spline args!
    # Very temporarily (just take line-spline):
    shift = 3
    n_knots = 2
    x_knots = np.linspace(min(X), max(X), n_knots)
    best_fit = np.zeros((2, (2 + n_knots)))
    for i, Y in enumerate(Y_data):
        a, b = np.polyfit(X, Y, deg=1)
        y_knots = a*x_knots + b
        best_fit[i, :] = np.concatenate([[0, 0.3], y_knots])

    # Perform maximum likelihood fit to get mu_hat, theta_hat
    # Parameter array: mu, theta_H1, theta_H2
    bounds = [(None, None)]
    for Y in Y_data:
        bounds += [(None, None), (1e-3, None)]  # alpha, sigma
        minY, maxY = min(Y), max(Y)
        bounds += [(minY, maxY) for y in y_knots]

    t0 = time.time()
    popt = minimize(lambda x: -likelihood(x, idx_peak, X, Y_data, x_knots, peak_norm, shift=shift),
                    x0=np.concatenate([[80], best_fit.flatten()]),
                    method="L-BFGS-B",
                    options={"ftol": 1e-12, "gtol": 1e-12},
                    bounds=bounds)
    t1 = time.time()
    tqdm.write(str(t1 - t0))

    # Now get sigma_mu_hat from dlnL/dmu
    Fisher = 0.
    for i, Y in enumerate(Y_data):
        y_knots = popt.x[1 + 2 + i*(2+n_knots):1 + (i+1)*(2+n_knots)]
        y_spline = CubicSpline(x_knots, y_knots, extrapolate=False)
        # TODO: adapt for non-delta peaks
        Lambda = Y[idx_peak] - y_spline(X[idx_peak])
        Fisher += second_derivative(
            popt.x[1+i*(2 + n_knots)],
            np.sqrt(popt.x[2+i*(2+n_knots)]),
            Lambda,
            popt.x[0]
        )

    # Return mu_hat, sigma_dlnL, bkg
    bkg = y_spline(X[idx_peak])  # TODO: adapt for non-delta peaks
    return popt.x[0], np.sqrt(-1. / Fisher), bkg


def process_segment(args):
    """Calculate an upper limit - wrapper for multiprocessing."""
    i, freq_subset, logPSD_subset, fi, lo, peak_norm = args
    return i, lo+fi, get_upper_limits_approx(freq_subset, logPSD_subset, fi, peak_norm)


def generate_noise(freq, x_knots, model_params):
    """Generate PSD based on skew-norm fits on real data."""
    fit_data = np.load("fit_results_skew.npz", allow_pickle=True)
    idx, interp = fit_data["idx"], fit_data["interp"]
    y_noise = np.zeros_like(freq)
    _params = np.zeros((len(idx), 2))
    for i, (start, end) in enumerate(idx):
        mu, sigma, alpha = list(map(lambda f: f(i), interp))
        y_noise[start:end] = skewnorm.rvs(alpha, loc=mu, scale=sigma, size=end-start)
        _params[i, :] = alpha, sigma
        if i == 0:
            first_start = start
        elif i == len(idx) - 1:
            last_end = end

    X, y_noise = freq[first_start:last_end], y_noise[first_start:last_end]
    y_model = models.model_spline(model_params, x_knots, shift=0)(X)
    Y = y_noise + y_model[first_start:last_end]
    return X, Y


def main():
    """Get all necessary data and launch analysis."""
    # Analysis variables
    segment_size = 1000
    pruning = 1000
    fmin = 10  # Hz
    fmax = 8192  # Hz
    resolution = 1e-6
    num_processes = 1
    # TODO relative
    calib_path = "../shared_git_data/Calibration_factor_A_star.txt"

    # Process variables
    Jdes = utils.Jdes(fmin, fmax, resolution)
    freq = np.log(np.logspace(np.log10(fmin), np.log10(fmax), Jdes))

    # Get model params (for generation)
    x_knots = np.array([2.30258509, 3.04941017, 3.79623525,
                        8.65059825, 8.95399594, 8.97733423, 9.02401079])
    model_params = np.load("bkg_model_params.npy")

    # Generate noise
    logPSD_data = [generate_noise(freq, x_knots, model_params)[1]
                   for i in range(2)]  # H1, L1

    # Calculate upper limits on generated noise data
    # 0. Get A_star
    calib = np.loadtxt(calib_path, delimiter="\t")
    f_calib = [interp1d(calib[:, 1], calib[:, 0])
               for i in range(2)]  # H1, L1  # TODO: other file
    # 0.1 Adapt frequency range to calibrated values
    freq = generate_noise(freq, x_knots, model_params)[0]
    mask = (freq >= np.log(min(calib[:, 1]))) & (freq <= np.log(max(calib[:, 1])))
    freq = freq[mask]
    logPSD_data = [logPSD[mask] for logPSD in logPSD_data]

    # 1. Generate args
    segment_borders = np.arange(0, len(freq), segment_size)  # Does not include the last frequencies
    def args():
        """Loop over segments and ifos to give data and args."""
        idx = 0
        for i in range(len(segment_borders) - 1):
            lo, hi = segment_borders[i], segment_borders[i+1]
            freq_subset = freq[lo:hi].copy()
            logPSD_subset = np.zeros((len(logPSD_data), hi-lo))
            for j, logPSD in enumerate(logPSD_data):  # H1, L1
                logPSD_subset[j, :] = logPSD[lo:hi].copy()

            n_frequencies = max(1, (hi - lo) // pruning)
            for fi in list(map(lambda x: int(np.round(x)),
                               np.linspace(0, hi-lo, n_frequencies))):
                # Calculate a normalisation constant to pass to the peak
                # This includes calibration and conversion
                freq_Hz = np.exp(freq_subset[fi])
                a_star_sqr = np.array([f(freq_Hz)**2 for f in f_calib])
                peak_norm = np.log(gamma / (a_star_sqr * np.exp(3*freq_subset[fi])))
                # These are the arguments to pass to a single computation
                yield (idx, freq_subset, logPSD_subset, fi, lo, peak_norm)
                idx += 1

    # 2. Create job Pool
    n_upper_limits = (len(segment_borders) - 1) // (segment_size // pruning)
    upper_limit_data = np.zeros((4, n_upper_limits))
    with Pool(num_processes) as p:
        results = list(tqdm(p.imap(process_segment, args()),
                            total=n_upper_limits, position=0, desc="Calculation"))

    # 3. Merge results
    for i, fi, upper_limit in results:
        upper_limit_data[0, i] = np.exp(freq[fi])
        upper_limit_data[1:, i] = upper_limit  # mu, sigma, bkg
    np.save("tmp.npy", upper_limit_data)

    # 4. Plot!
    plt.figure()
    n = upper_limit_data.shape[1]
    coupling = np.exp(-0.5*upper_limit_data[1, :n])
    sigma = 0.5*coupling*upper_limit_data[2, :n]

    ax = plt.subplot(111)
    ax.set_yscale("log")
    ax.errorbar(upper_limit_data[0, :n], coupling, sigma, fmt=".")
    plt.show()


if __name__ == '__main__':
    main()
