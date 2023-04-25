import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit as fit
from scipy.stats import skewnorm
from scipy.special import lambertw


def gaus(x, A, mu, sigma):
    return A / (np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*((x - mu) / sigma)**2)

def skew_gaus(x, A, mu, sigma, skew):
    return A*skewnorm.pdf(x, a=skew, loc=mu, scale=sigma)

def log_gaus(x, A, mu, sigma):
    return A / (np.sqrt(2*np.pi)*sigma*x) * np.exp(-0.5*((np.log(x) - mu) / sigma)**2)

def reverse_log_gaus(x, A, mu, sigma, right=True):
    """
    Calculate the reverse log gausian.
    
    right=True to get the value on the right side of the peak.
    Set False for the left side.
    """
    
    sqrt_term = sigma**2 - 2.*(mu + np.log(x*np.sqrt(2*np.pi)*sigma/A))
    if right:
        return np.exp(mu - sigma**2 + sigma*np.sqrt(sqrt_term))
    return np.exp(mu - sigma**2 - sigma*np.sqrt(sqrt_term))

def log_skew_gaus(x, A, mu, sigma, skewness):
    return A*skewnorm.pdf(np.log(x), a=skewness, loc=mu, scale=sigma)

def piecewise_log_gaus(x, sigma1, alpha, B, mu2, sigma2):
    alpha, B = np.abs(alpha), np.abs(B)  # Must be positive
    mu1 = (mu2 - sigma2**2 - np.log(alpha))*(sigma1 / sigma2)**2 + sigma2**2 + np.log(alpha)
    A = B*sigma1/sigma2 * np.exp(-0.5*(((np.log(alpha) - mu2) / sigma2)**2 - ((np.log(alpha) - mu1) / sigma1)**2))
    
    mask_right = x >= alpha
    output = np.zeros_like(x)
    output[~mask_right] = log_gaus(x[~mask_right], A, mu1, sigma1)
    output[mask_right] = log_gaus(x[mask_right], B, mu2, sigma2)
    return output


def fit_hist(f, data, bins, p0=None):
    """
    Proper chi-square fit.
    
    f: function to be used in the fit.
    data: data to be histogrammed in bins (passed to np.histogram).
    p0: starting parameters for the fit. If not given std. gaus params will be applied.
    """
    h0, bin_edges = np.histogram(data, bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = bin_edges[1:] - bin_edges[:-1]
    
    # First fit that ignores zero-content bins
    # If there are no zeros, jump to the second fit
    if p0 is None:  # Assume gaussian
        p0 = [10.*max(h0), bin_centers[np.argmax(h0)], np.std(data, ddof=1)]

    m = h0 == 0
    if np.sum(np.array(m, dtype=int)):
        zero_h0, zero_bin_centers, zero_bin_width = h0[~m], bin_centers[~m], bin_width[~m]
        p0, _ = fit(f, zero_bin_centers, zero_h0/zero_bin_width, p0=p0,
                    sigma=np.sqrt(zero_h0)/zero_bin_width, absolute_sigma=True)
        h0 = np.array(h0, dtype=np.float64)
        h0[m] = f(bin_centers[m], *p0) * bin_width[m]
    
    # Second fit using output of previous one
    popt, pcov = fit(f, bin_centers, h0/bin_width, p0=p0,
                     sigma=np.sqrt(h0)/bin_width, absolute_sigma=True)
    
    return popt, pcov


def plot_hist(data, bin_edges, ax=None, range=None, logx=False, logy=False):
    """Plot a histogram, return ax."""
    if ax is None:
        plt.figure()
        ax = plt.subplot(111)
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    if range is None:
        # Assume bins all have the same size
        bin_width = bin_edges[1] - bin_edges[0]
        range = (min(bin_centers) - bin_width, max(bin_centers) + bin_width)
    
    # Plot histogram with Poisson uncertainty
    ax.hist(data, bin_edges, range=range, histtype='step', linewidth=2)
    h0, _ = np.histogram(data, bin_edges)
    ax.errorbar(bin_centers, h0, yerr=np.sqrt(h0), fmt='none', color='black')
    
    # Style & return
    ax.grid(color="grey", linestyle="--", linewidth=1.5, alpha=.5)
    return ax
    
def plot_func_hist(f, args, data, bin_edges, ax=None,
                   logy=False, logx=False, range=None, color=None):
    """Plot a function through a histogram."""
    ax = plot_hist(data, bin_edges, ax=ax, logy=logy, logx=logx, range=range)
    
    # Plot function
    x = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = bin_edges[1:] - bin_edges[:-1]
    ax.plot(x, f(x, *args) * bin_width, color=color)
    
    # TODO: plot residuals
    return ax
