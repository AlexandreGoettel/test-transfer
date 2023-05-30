import numpy as np
from scipy.stats import norm
from scipy.special import erf, owens_t


def get_nth_letter(i):
    """Return the ith letter of the alphabet in lowercase."""
    if 1 <= i <= 26:
        return chr(ord('a') + i - 1)
    return "Invalid input"


def read(filename, n_lines=None):
    """Read LPSD output - return frequencies, PSD."""
    x, y = [], []
    with open(filename, 'r') as _file:
        data = csv.reader(_file, delimiter="\t")
        for row in tqdm(data, total=n_lines):
            if not row or row[0].startswith("#"):
                continue
            x += [float(row[0])]
            y += [float(row[1])]

    return x, y


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
