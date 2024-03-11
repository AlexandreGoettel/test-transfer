"""Statistical tools for the LPSD analysis."""
import numpy as np
from scipy.stats import norm, skewnorm, skew
from scipy.special import owens_t, erf
from scipy.signal import convolve


_LOG_2_PI = 0.5 * np.log(2 * np.pi)
_LOG_2_PI_TERM = np.log(2) - _LOG_2_PI


def pdf_skewnorm(x, A, loc=0, scale=1, alpha=0):
    """Wrapper for skew-normal amplitude with amplitude A."""
    return A*skewnorm.pdf(x, a=alpha, loc=loc, scale=scale)


def cdf_skewnorm(x, loc=0, scale=1, alpha=0):
    """Normalised decentralised skew normal CDF"""
    z = (x - loc) / scale
    return norm.cdf(z) - 2 * owens_t(z, alpha)


# def logpdf_skewnorm(x, loc=0, scale=1, alpha=0):
#     """Implement skewnorm.logpdf for speed."""
# FIXME: this function does NOT work properly
#     norm_term = -0.5*((x - loc) / scale)**2
#     lin_skew_term = 0.5*(1. + erf(alpha*(x - loc) / (np.sqrt(2)*scale)))
#     lin_skew_term[lin_skew_term == 0] = np.nan
#     skew_term = np.log(lin_skew_term)
#     return _LOG_2_PI_TERM + norm_term + skew_term


def logpdf_norm(x, loc=0, scale=1):
    """Implement Gaus logPDF."""
    return -_LOG_2_PI - np.log(scale) - 0.5*((x - loc) / scale)**2

def get_skewnorm_p0(data, max_alpha=5):
    """Get estimate alpha, mu, sigma from data."""
    # Get moments from data
    # https://modelingwithdata.org/pdfs/moments.pdf
    n = len(data)
    mean = np.mean(data)
    var = ((data - mean)**2).sum() / (n - 1)
    _skew = ((data - mean)**3).sum() * n / ((n - 1) * (n - 2))
    _skew = skew(data, bias=False)

    # Correct for numerical problems
    if np.abs(_skew) > .99:
        _skew = np.sign(_skew)*.99

    # Get skew normal parameters
    gamma = np.abs(_skew)**(2/3.)
    delta = np.sign(_skew) * np.sqrt(0.5*np.pi*gamma / (gamma + (0.5*(4 - np.pi))**(2/3.)))
    alpha = delta / np.sqrt(1. - delta**2)
    sigma = np.sqrt(var / (1. - 2/np.pi*delta**2))
    mu = mean - sigma * delta * np.sqrt(2 / np.pi)

    # Correct for "unphysicalness"
    alpha = np.sign(alpha)*max_alpha if abs(alpha) > max_alpha else alpha

    return alpha, mu, sigma


def kde_smoothing(timeseries, kernel_size):
    """Smooth a 1-D ndarray using a Gaussian kernel."""
    kernel_bound = int(kernel_size) + 1
    kernel_range = np.linspace(-kernel_bound, kernel_bound, 2 * kernel_bound + 1)
    kernel = norm.pdf(kernel_range, scale=kernel_size / 3)  # Scale is roughly 1/3 of kernel size
    kernel /= kernel.sum()  # Normalize the kernel

    # Reflect the time series at both ends & convolve
    padded_timeseries = np.concatenate((timeseries[:kernel_bound][::-1],
                                        timeseries,
                                        timeseries[-kernel_bound:][::-1]))
    smoothed_timeseries = convolve(padded_timeseries, kernel, mode='valid')
    return smoothed_timeseries
