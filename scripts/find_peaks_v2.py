import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm, skewnorm
from tqdm import trange
from scipy.optimize import curve_fit as fit
# Project imports
from peakFinder import PeakFinder
import hist
import utils


def get_skew_norm_mode(mu, sigma, alpha):
    """Return an approximate formula for the mode of a skew Gaussian."""
    delta = alpha / np.sqrt(1. + alpha**2)
    muz = np.sqrt(2. / np.pi)*delta
    gamma = (4. - np.pi)/2.*(muz)**3 / (1 - muz**2)**(1.5)
    mode = muz - gamma*np.sqrt(1. - muz**2)/2 - np.sign(alpha)/2.*np.exp(-2*np.pi/np.abs(alpha))
    return mode*sigma + mu


def get_skew_norm_var(alpha, sigma):
    """Return the variance of a skew normal with shape parameter alpha."""
    delta = alpha / np.sqrt(1. + alpha**2)
    return sigma**2*(1. - delta**2*2/np.pi)


def distr_var(x, y):
    """Return the variance of the distribution."""
    norm = np.sum(y)
    mean = np.sum(x*y) / norm
    variance = np.sum(y * (x - mean)**2) / norm
    return variance


# TODO: extract settings from nohup.out
def main():
    """Find peaks in a block-approximation output file."""
    # Configure data
    fs = 16384  # Hz
    fmin = 10  # Hz
    fmax = 8192  # Hz
    resolution = 1e-6
    epsilon = 0.1
    filename = "../o3-lpsd-results/result_epsilon10_1243393026_1243509654_L1.txt"
    
    # Get data & block positions
    pf = PeakFinder(fs=fs, resolution=resolution, fmin=fmin, fmax=fmax, epsilon=epsilon, name=filename)
    positions = np.fromiter(pf.block_position_gen(), dtype=int)
    
    # Configure analysis
    nbins = 400
    n_sigma = 4
    n_poly = 3
    max_chi = 9
    
    chi_sqr = np.zeros(len(positions) - 1)
    # For each block
    results = np.zeros_like(pf.psd)
    PSD_clean = np.zeros_like(pf.psd)
    max_block_id = 49  # because it is difficult to fit high-peak density blocks
    for i_segment in trange(len(positions) - 1):  
        if i_segment > max_block_id:  # Ignore last blocks
            continue
        # Fit with a simple skewed gaussian
        validFit = True
        try:
            popt, _, _chi_sqr = pf.baseline_fit(positions[i_segment], positions[i_segment+1],
                                                nbins)
        except RuntimeError:
            _chi_sqr = np.inf
            validFit = False
            
        if not validFit or _chi_sqr > 20:
            # If that doesn't work, try new fit with cutoff around peak as protection
            try:
                popt, _, _chi_sqr = pf.baseline_fit_cutoff(positions[i_segment], positions[i_segment+1],
                                                            nbins, cutoff=0.1)
            except RuntimeError:
                # Abandon this segment
                continue
        
        # If chi_sqr is still too bad, abandon segment
        chi_sqr[i_segment] = _chi_sqr
        if _chi_sqr > 50:
            continue
        
        # Define boundary based on fit
        sigma_value = 2.*norm.cdf(n_sigma) - 1.
        _, mu, sigma, skew = popt
        boundary = skewnorm.ppf(sigma_value, a=skew, loc=mu, scale=sigma)
        
        # Apply boundary on PSD data
        x = pf.freq[positions[i_segment]:positions[i_segment+1]]
        y = pf.psd[positions[i_segment]:positions[i_segment+1]]
        mask = np.log(y) < boundary
        xF, yF = x[mask], y[mask]
        
        # Fit poly through baseline baseline
        def poly(x, *args, deg=2):
            output = np.zeros_like(x)
            for i in range(deg):
                output += x**(deg-i)*args[i]
            return output
        
        def fitFunc(x, *args):
            return poly(x, *args, deg=n_poly)
        
        # Perform fit in log space (numerically easier)
        ylog = np.log(yF)
        p0 = np.polyfit(xF, ylog, deg=n_poly)
        popt, pcov = fit(fitFunc, xF, ylog, p0)
        del xF, yF, ylog
        
        # Save fit results
        results[positions[i_segment]:positions[i_segment+1]] = np.exp(fitFunc(x, *popt))
        
        # continue
        # ######################################### #
        # Step 2 - Fit subtracted projection        #
        # ######################################### #
        # Fit and calculate chi square
        yClean = y - results[positions[i_segment]:positions[i_segment+1]]

        # Try to fit a skewed gaussian
        # If that doesn't work, calculate sigma from quantiles
        # Check if mode from skew gaussian is in manually calculated quantiles
        # Also place a cut on the chi square value

        # Calculate mode, sigma from quantiles
        _lo, _hi = np.quantile(yClean, [0.16, 0.84])

        # Prepare fit
        f = hist.skew_gaus
        # Make sure bins are fine enough
        _nbins = max(nbins, int((max(yClean) - min(yClean)) / (.5*(_hi-_lo))+.5))
        _nbins = min(_nbins, int(5e4))  # Just to make sure
        bins, bin_centers, bin_width = utils.getBinVars(yClean, _nbins, log=False)
        h0, _ = np.histogram(yClean, bins)
        try:
            _argmax = np.argmax(h0)
            p0 = [h0[_argmax]*10, bin_centers[_argmax], np.std(yClean, ddof=1), 0]
            popt, pcov = hist.fit_hist(f, yClean, bins, p0=p0)
            chi_sqr[i_segment] = utils.getChiSquarePoisson(f(bin_centers, *popt)*bin_width, h0)
        except RuntimeError:
            popt, pcov = None, None
            chi_sqr[i_segment] = np.inf
        
        # Place cuts on fit results, use quantiles instead if they fail
        if popt is None:
            # mu, sigma = bin_centers[np.argmax(h0)], (_hi - _lo) / 2.
            mu, sigma = np.median(yClean), (_hi - _lo) / 2.
        else:
            mode = get_skew_norm_mode(popt[1], popt[2], popt[3])
            if mode < _lo or mode > _hi\
                or chi_sqr[i_segment] > max_chi:
                    # mu, sigma = bin_centers[np.argmax(h0)], (_hi - _lo) / 2.
                    mu, sigma = np.median(yClean), (_hi - _lo) / 2.
            else:
                mu, sigma = mode, np.sqrt(get_skew_norm_var(popt[3], popt[2]))

        # Do a second fit with a cut on the sigma distance to the peak
        bins, bin_centers, bin_width = utils.getBinVars([mu-5*sigma, mu+5*sigma], nbins, log=False)
        h0, _ = np.histogram(yClean, bins)
        _argmax = np.argmax(h0)
        p0 = [h0[_argmax]*10, bin_centers[_argmax], np.sqrt(distr_var(bin_centers, h0)), 0]
        popt, pcov = hist.fit_hist(f, yClean, bins, p0=p0)
        chi_sqr[i_segment] = utils.getChiSquarePoisson(f(bin_centers, *popt)*bin_width, h0)
        
        # Re-calculate
        mode = get_skew_norm_mode(popt[1], popt[2], popt[3])
        if mode < _lo or mode > _hi\
            or chi_sqr[i_segment] > max_chi:
                mu, sigma = bin_centers[np.argmax(h0)], (_hi - _lo) / 2.
        else:
            mu, sigma = mode, np.sqrt(get_skew_norm_var(popt[3], popt[2]))
        
        # tmp - plots
        p1, p2 = positions[i_segment], positions[i_segment+1]
        if not np.isinf(chi_sqr[i_segment]):
            ax = hist.plot_func_hist(f, popt, yClean, bins)
            ax.set_title(f"{i_segment}:{chi_sqr[i_segment]:.2f}, {pf.freq[p1]:.1f}-{pf.freq[p2]:.1f}Hz")
            ax.axvline(mu, color="r", linewidth=2.5)
            ax.axvline(_lo, color="g", linestyle="--")
            ax.axvline(_hi, color="g", linestyle="--")
            plt.show()

        # Whiten block
        p1, p2 = positions[i_segment], positions[i_segment+1]
        PSD_clean[p1:p2] = (yClean - mu) / sigma
        del yClean
    
    # Plot chi-sqr
    plt.figure()
    plt.plot(chi_sqr)
    x = np.arange(len(chi_sqr))
    plt.scatter(x, chi_sqr)
    plt.title(r"$\chi^2/ndof$")
    
    # Prepare to plot results overlapped with PSD
    n = 100
    x = pf.freq[::n]
    y = pf.psd[::n]
    y_model = results[::n]
    fig = plt.figure()
    gs = gridspec.GridSpec(7, 1)
    ax, axLog, axRes = fig.add_subplot(gs[:3]), fig.add_subplot(gs[3:6]),fig.add_subplot(gs[6])
    
    # Plot
    ax.plot(x, y, label="Data")
    ax.plot(x, y_model, label=f"Block poly={n_poly}")
    ax.plot(x, np.abs(y - y_model), label="Subtracted model", alpha=.2)
    axLog.plot(x, np.abs(y - y_model), color="C2")
    axRes.plot(x, y - y_model)
    
    # Nice things
    ax.set_yscale("log")
    ax.set_xscale("log")
    axLog.set_xscale("log")
    axLog.set_yscale("log")
    axRes.set_xscale("log")
    axRes.set_xlabel("Frequency (Hz)")
    ax.set_xticks([], [])
    axLog.set_xticks([], [])
    ax.set_ylabel("PSD")
    axLog.set_ylabel("|Data - Model|")
    axRes.set_ylabel("Residual")
    ax.legend(loc="upper right")
    ax.grid(color="grey", alpha=.5, linestyle="--", linewidth=1.5)
    axLog.grid(color="grey", alpha=.5, linestyle="--", linewidth=1.5)
    axRes.grid(color="grey", alpha=.5, linestyle="--", linewidth=2.)
    
    xmin, xmax = pf.freq[positions[0]], pf.freq[positions[max_block_id+1]]
    ax.set_xlim(xmin, xmax)
    axRes.set_xlim(xmin, xmax)
    axLog.set_xlim(xmin, xmax)
    axRes.set_ylim(-1e-46, 1e-46)
    ax.set_ylim(1e-52, 1e-38)
    axLog.set_ylim(1e-52, 1e-38)
    axRes.axhline(0, color="r", linestyle="--", linewidth=2.)
    ax.set_title("Data cleaning")
    
    # Now plot "final" product
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, PSD_clean[::n])
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Cleaned PSD")
    plt.show()
    

if __name__ == '__main__':
    main()


# Notes:
# - to find small peaks, use Michaela's method
# - ignore last few segments. Focus on low F