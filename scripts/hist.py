import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit as fit
from scipy.stats import skewnorm


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
    A = B*sigma1/sigma2 * np.exp(-0.5*(((np.log(alpha) - mu2) / sigma2)**2 -\
        ((np.log(alpha) - mu1) / sigma1)**2))

    mask_right = x >= alpha
    output = np.zeros_like(x)
    output[~mask_right] = log_gaus(x[~mask_right], A, mu1, sigma1)
    output[mask_right] = log_gaus(x[mask_right], B, mu2, sigma2)
    return output


def fit_hist(f, data, bins, p0=None, get_chi_sqr=False, isHist=False):
    """
    Proper chi-square fit.
    
    f: function to be used in the fit.
    data: data to be histogrammed in bins (passed to np.histogram).
    p0: starting parameters for the fit. If not given std. gaus params will be applied.
    """
    # TODO: do N-independent fit!
    h0, bin_edges = (data, bins) if isHist else np.histogram(data, bins)
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

    if not get_chi_sqr:
        return popt, pcov

    # Protect against strongly decreasing functions
    func_values = f(bin_centers, *popt)
    mask = func_values != 0
    h0, func_values, bin_width = h0[mask], func_values[mask], bin_width[mask]
    chi_sqr = np.sum(
        np.power(h0 - bin_width*func_values, 2) / (bin_width*func_values)
    )
    return popt, pcov, chi_sqr / (len(h0) - len(popt))


def plot_hist(data, bin_edges, ax=None, xrange=None, do_errorbar=False,
              logx=False, logy=False, color=None, **kwargs):
    """Plot a histogram, return ax."""
    if ax is None:
        plt.figure()
        ax = plt.subplot(111)
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    if type(bin_edges) == int:
        bin_edges = np.linspace(min(data), max(data), bin_edges)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    if xrange is None:
        # Assume bins all have the same size
        bin_width = bin_edges[1] - bin_edges[0]
        xrange = (min(bin_centers) - bin_width, max(bin_centers) + bin_width)

    # Plot histogram with Poisson uncertainty
    ax.hist(data, bin_edges, range=xrange, histtype='step',
            linewidth=2, color=color, **kwargs)
    if do_errorbar:
        h0, _ = np.histogram(data, bin_edges)
        ax.errorbar(bin_centers, h0, yerr=np.sqrt(h0), fmt='none', color='black')

    # Style & return
    ax.grid(color="grey", linestyle="--", linewidth=1.5, alpha=.5)
    return ax

def plot_func_hist(f, args, data, bin_edges, logy=False, logx=False,
                   xrange=None, color=None, density=False, **kwargs):
    """Plot a function through a histogram."""
    # Prepare gridspec
    fig = plt.figure()
    gs = GridSpec(4, 1)
    ax, axRes = fig.add_subplot(gs[:3]), fig.add_subplot(gs[3])

    # Plot fit results function
    ax = plot_hist(data, bin_edges, ax=ax, logy=logy, logx=logx,
                   xrange=xrange, density=density)
    x = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = bin_edges[1:] - bin_edges[:-1]
    ax.plot(x, f(x, *args) * bin_width, color=color, **kwargs)

    # Plot residuals (not for zeros)
    h0, _ = np.histogram(data, bin_edges)
    m = h0 > 0
    _x, _h0, _bin_width = x[m], h0[m], bin_width[m]
    axRes.errorbar(_x, _h0 - f(_x, *args)*_bin_width, np.sqrt(_h0),
                   fmt=".", color=color, zorder=1)
    axRes.axhline(0, linewidth=2., linestyle="--", color="r", zorder=2)

    return ax, axRes
