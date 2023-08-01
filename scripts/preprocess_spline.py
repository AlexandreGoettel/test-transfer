"""Pre-process PSDs by (BF-optimized) Fitting splines through segments."""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import skewnorm
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
# Custom imports
import ultranest
# Project imports
from peak_finder import PeakFinder
import sensutils
import hist


def process(filename, ana_fmin=10, ana_fmax=8192, k_init=5, verbose=False, **kwargs):
    """Perform pre-processing on given LPSD output file."""
    kwargs["name"] = filename
    segment_size = kwargs.pop("segment_size")
    pf = PeakFinder(**kwargs)

    # 0. Separate in segments
    idx_start = np.where(pf.freq >= ana_fmin)[0][0]
    idx_end = np.where(pf.freq <= ana_fmax)[0][-1]
    positions = np.concatenate([
        np.arange(idx_start, idx_end, segment_size),
        [idx_end]])
    for start, end in zip(positions[:-1], positions[1:]):
        x, y = np.log(pf.freq[start:end]), np.log(pf.psd[start:end])
        # 1. Set initial guess along straight line
        a, b = np.polyfit(x, y, 1)
        x_knots = np.linspace(x[0], x[-1], k_init)

        # 2. Find initial parameters using scipy.minimize
        def fitFunc(x, A, mu, alpha, sigma):
            return A*skewnorm.pdf(x, alpha, scale=np.sqrt(sigma), loc=mu)

        res = y - CubicSpline(x_knots, a*x_knots+b, extrapolate=False)(x)
        bins = np.linspace(min(res), max(res), 100)
        popt, pcov = hist.fit_hist(fitFunc, res, bins, p0=[len(x), 0, 0, 1])
        # print(popt)
        # print(np.sqrt(np.diag(pcov)))
        # axes = hist.plot_func_hist(fitFunc, popt, res, bins)
        # plt.show()

        def lkl(params):
            [alpha, sigma], y_knots = params[:2], params[2:]
            y_model = CubicSpline(x_knots, y_knots, extrapolate=False)(x)
            out = skewnorm.logpdf(y - y_model,
                                  alpha,
                                  loc=-sensutils.get_mode_skew(0, np.sqrt(np.abs(sigma)), alpha),
                                  scale=np.sqrt(np.abs(sigma)))
            out[np.isinf(out)] = 10000
            return -out.sum() / len(x)

        initial_guess = np.concatenate(
            [[popt[2], popt[3]], a*x_knots + b +
             sensutils.get_mode_skew(popt[1], np.sqrt(popt[3]), popt[2])])
        bounds = [(-10, 10), (.1, None)]
        bounds += [(min(y), max(y)) for x in x_knots]
        popt = minimize(lkl, x0=initial_guess, method="L-BFGS-B",
                        bounds=bounds)

        print(popt)
        print(initial_guess)
        # popt.x = initial_guess
        print(lkl(popt.x))
        print(lkl(initial_guess))

        y_model = CubicSpline(x_knots, popt.x[2:])(x)#a*x_knots+b)(x)#popt.x[2:])(x)
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, y_model)
        res = y - y_model
        bins = np.linspace(min(res), max(res), 50)
        bin_centers = (bins[1:] + bins[:-1]) / 2.
        ax = hist.plot_hist(res, bins, density=True)

        alpha_skew, sigma_skew = popt.x[0], np.sqrt(np.abs(popt.x[1]))
        mode = -sensutils.get_mode_skew(0, sigma_skew, alpha_skew)
        ax.plot(bin_centers, skewnorm.pdf(bin_centers, alpha_skew, scale=sigma_skew, loc=mode))
        plt.show()
        return

        # 3. Start Bayesian optimisation procedure around optimal point
        # 4. Save to file
        pass


def main():
    """Organise analysis."""
    # TODO: rel. path
    path = "data/result_epsilon10_1243393026_1243509654_H1.txt"
    kwargs = {"epsilon": 0.1,
              "fmin": 10,
              "fmax": 8192,
              "fs": 16384,
              "resolution": 1e-6
              }
    process(path, segment_size=10000, ana_fmin=10, ana_fmax=5000, **kwargs)


if __name__ == '__main__':
    main()
