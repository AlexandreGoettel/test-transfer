import csv
from tqdm import tqdm
import numpy as np
from scipy.stats import norm
from scipy.special import erf, owens_t


def seq_mean(x, y):
    """Calculate the mean over a sequence (e.g. hist.)"""
    return np.sum(x * y) / np.sum(y)


def seq_var(x, y):
    """Calculate the variance over a sequence (e.g. hist.)"""
    mu = seq_mean(x, y)
    return np.sum(y*(x - mu)**2) / np.sum(y)


def get_nth_letter(i):
    """Return the ith letter of the alphabet in lowercase."""
    if 1 <= i <= 26:
        return chr(ord('a') + i - 1)
    return "Invalid input"


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


def read(name, dtype=np.float64, n_lines=None,
         delimiter="\t", raw_freq=True):
    """
    Read an output file from LPSD.

    name(str): output file name.
    return: frequency & PSD arrays.
    """
    x, y = [], []
    with open(name, "r") as _file:
        data = csv.reader(_file, delimiter=delimiter)
        print("Reading input PSD..")
        for row in tqdm(data, total=n_lines):
            try:
                if raw_freq:
                    x += [float(row[0])]
                y += [float(row[1])]
            except (ValueError, IndexError):
                continue
    return np.array(x, dtype=dtype), np.array(y, dtype=dtype)


def skew_normal_cdf(x, mu=0, sigma=1, alpha=0):
    z = (x - mu) / sigma
    return norm.cdf(z) - 2 * owens_t(z, alpha)


def f(j, fmin, fmax, Jdes):
    g = np.log(fmax) - np.log(fmin)
    return fmin * np.exp(j*g / (Jdes - 1))


def r_prime(j, fmin, fmax, Jdes):
    g = np.log(fmax) - np.log(fmin)
    return f(j, fmin, fmax, Jdes) * (np.exp(g / (Jdes - 1)) - 1.)


def Jdes(fmin, fmax, resolution):
    g = np.log(fmax) - np.log(fmin)
    return int(np.floor(1 + g / np.log(1. + resolution)))


def N(j, fmin, fmax, fs, resolution):
    g = np.log(fmax) - np.log(fmin)
    J = Jdes(fmin, fmax, resolution)
    return fs/fmin * np.exp(-j*g / (J - 1.)) / (np.exp(g / (J - 1.)) - 1.)


def m(fmin, fmax, resolution):
    g = np.log(fmax) - np.log(fmin)
    J = Jdes(fmin, fmax, resolution)
    return 1. / (np.exp(g / (J - 1.)) - 1.)

def j(j0, J, g, epsilon, fmin, fmax, fs, resolution):
    Nj0 = N(j0, fmin, fmax, fs, resolution)
    return - (J - 1.) / g * np.log(Nj0*(1 - epsilon) * fmin/fs * (np.exp(g / (J - 1.)) - 1.))


def getBinVars(y, nbins, log=True):
    if log:
        bins = np.logspace(np.log10(min(y)), np.log10(max(y)), nbins, base=10)
    else:
        bins = np.linspace(min(y), max(y), nbins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.
    bin_width = bins[1:] - bins[:-1]  # not always a constant!    
    return bins, bin_centers, bin_width


def getChiSquarePoisson(y, h, ddof=3):
    """Get reduced chi square distance between the two arrays by assuming that x follows Poisson."""
    zero = h == 0
    return np.sum(np.power(y[~zero] - h[~zero], 2) / h[~zero]) / (len(h[~zero]) - ddof)


def calc_fwhm_pos(x, y):
    """Calculates the Full Width at Half Maximum (FWHM) of a curve defined by the `x` and `y` data.
    
    Args:
        x (ndarray): The x-values of the curve.
        y (ndarray): The y-values of the curve.
    
    Returns:
        The positions that encompass the FWTHM
    """
    # Calculate the half-maximum value of the curve
    hm = np.max(y) / 2
    
    # Find the indices of the first and last points above half-maximum
    idx_start = np.argwhere(y > hm)[0][0]
    idx_end = np.argwhere(y > hm)[-1][0]
    
    # Calculate the FWHM
    return x[idx_start], x[idx_end]
