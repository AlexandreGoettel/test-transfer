"""Calculate experimental sensitivity based on simulated data."""
from tqdm import trange, tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.stats import skewnorm
from scipy.special import erfc
from scipy.signal import convolve
import ultranest
# Project imports
import models
import utils
import hist


def run_simple_spline_fit(X, Y, x_knots, verbose=False):
    """
    Use ultranest to run sampling over given x_knots, return (log)evidence.
    
    Note: fit in log-log space!
    """
    def model(cube, x_knots, shift=0):
        y_knots = cube[shift:len(x_knots)+shift]
        return CubicSpline(x_knots, y_knots, extrapolate=False)

    def likelihood(cube, x_data, y_data, x_knots, shift=3):
        y_model = model(cube, x_knots, shift=shift)(x_data)
        lkl = skewnorm.logpdf(y_data - y_model,
                              cube[0],
                              loc=0,
                              scale=np.sqrt(np.abs(cube[1])))
        # Take care of inf values
        lkl_inf = np.isinf(lkl)
        n_inf = np.sum(np.array(lkl_inf), dtype=int)
        if n_inf == len(lkl):
            return -1e4 * len(lkl)  # empirical
        if n_inf:
            # This will give theoretically incorrect values
            # But only in cases to-be-rejected anyways
            lkl[lkl_inf] = np.min(lkl[~lkl_inf])
        return np.sum(lkl)

    def prior(cube, initial_guess, deviation=1.1):
        params = cube.copy()

        bounds = [(0, 10), (.001, 10)]  # alpha, sigma
        for i, (lo, hi) in enumerate(bounds):
            params[i] = lo + cube[i] * (hi - lo)
        for i, mu in enumerate(initial_guess[shift:]):
            lo, hi = mu / deviation, mu * deviation
            params[i+shift] = lo + cube[i+shift] * (hi - lo)
        return params

    def _likelihood(cube, *_):
        return likelihood(cube, X, Y, x_knots, shift)

    def _prior(cube, *_):
        return prior(cube, initial_guess)

    # Define parameters: mu, sigma for Rayleigh, then spline
    parameters = [r"$\alpha$", r"$\sigma^2$"]
    shift = len(parameters)
    for i in range(len(x_knots)):
        parameters += [f"$k_{i}$"]

    # Estimate starting parameters (useful for prior)
    # i) Estimate very broadly from data
    initial_guess = np.zeros(len(parameters))
    for i, knot in enumerate(x_knots):
        jlim = np.where(knot >= X)[0][0]
        initial_guess[i+shift] = np.median(Y[max(0, jlim-50):jlim+50])
    initial_guess[:shift] = 2, 1

    # ii) Refine estimate with minimize, then pass that result to prior
    bounds = [(0, 10), (.1, 10)] + [(min(Y), max(Y)) for i in x_knots]
    popt = minimize(lambda x: -_likelihood(x), initial_guess, method="L-BFGS-B",
                    bounds=bounds, tol=1e-4)
    initial_guess = popt.x

    # Run UltraNest
    sampler = ultranest.ReactiveNestedSampler(
        parameters, _likelihood, transform=_prior,
        vectorized=False, resume="overwrite"
    )
    results = sampler.run(viz_callback=False,
                          min_num_live_points=128)

    if verbose:
        # Plot results
        print(popt)
        sampler.print_results()
        bf = results["maximum_likelihood"]["point"]

        plt.figure()
        plt.plot(X, Y, zorder=1)
        plt.plot(X, model(bf, x_knots, shift=shift)(X), label="ultranest", zorder=2)
        plt.plot(X, model(popt.x, x_knots, shift=shift)(X), label="minimize", zorder=2)
        plt.legend(loc="best")

        # Plot residuals
        residuals = Y - model(bf, x_knots, shift=shift)(X)
        bins = np.linspace(min(residuals), max(residuals), 100)

        def mdistr(x, alpha, scale):
            return skewnorm.pdf(x, alpha, loc=0, scale=scale) * len(residuals)
        hist.plot_func_hist(mdistr, bf[:shift], residuals, bins)
        plt.show()
    return results


def optimise_x_knots(X, Y, k=5, verbose=False):
    """Recursively optimise a spline to fit the data with a large evidence."""
    # Iterate over sampling runs
    x_knots = np.linspace(min(X), max(X), k)
    results = run_simple_spline_fit(X, Y, x_knots, verbose=verbose)
    evidence = results["logz"]
    print(f"Starting evidence: {evidence:.1f}")

    new_ev = -np.inf
    while True:
        # Delete knot?
        ev = np.zeros(len(x_knots) - 2)
        for i, idx in enumerate(trange(1, len(x_knots)-1,
                                       desc="Delete?")):
            _x_knots = np.delete(x_knots, idx)
            ev[i] = run_simple_spline_fit(X, Y, _x_knots, verbose=False)["logz"]
        idx_del = np.argmax(ev)
        new_ev = ev[idx_del]

        # Create knot?
        ev = np.zeros(len(x_knots) - 1)
        for i, idx in enumerate(trange(len(x_knots) - 1, desc="Create?")):
            _x_knots = np.insert(x_knots, idx+1, np.mean(x_knots[idx:idx+2]))
            ev[i] = run_simple_spline_fit(X, Y, _x_knots, verbose=False)["logz"]

        # Book-keeping
        idx_crt = np.argmax(ev)
        if ev[idx_crt] > new_ev:
            new_ev = ev[idx_crt]

        if new_ev <= evidence + .5:  # Has to exceed threshold
            break
        elif new_ev == ev[idx_crt]:
            x_knots = np.insert(x_knots, idx_crt+1, np.mean(x_knots[idx_crt:idx_crt+2]))
        else:
            x_knots = np.delete(x_knots, idx_del + 1)

        # Loop condition
        print(evidence, new_ev)
        # evidence = new_ev
        # tmp
        results = run_simple_spline_fit(X, Y, x_knots, verbose=True)
        evidence = results["logz"]
        if len(x_knots) == 2:
            break
    print("Final evidence: ", run_simple_spline_fit(X, Y, x_knots, verbose=True)["logz"])
    return x_knots, results["maximum_likelihood"]["point"]


def get_upper_limits(X, Y, pruning=1):
    """Use the cowan MC method with custom likelihood to estimate the q_mu upper lim."""
    # For now, use Rayleigh distribution in the likelihood, use skewnorm on real data
    # 0) Optimise spline for background model
    # x_knots, best_fit = optimise_x_knots(X[::pruning], Y[::pruning], verbose=True)
    # TMP
    # Do polyfit # TODO: then fit projection    
    # best_fit = np.array([2.19, 1.09, -92.41, -92.43])
    x_knots = np.linspace(min(X), max(X), 2)
    a, b = np.polyfit(X, Y, deg=1)
    y_knots = a*x_knots + b
    best_fit = [2, 1, y_knots[0], y_knots[1]]

    # Define likelihood
    def model(cube, idx_peak, x_data, x_knots, shift=3):
        # Spline background + single peak model (one-bin peak)
        y_knots = cube[shift:len(x_knots)+shift]
        y_spline = CubicSpline(x_knots, y_knots, extrapolate=False)(x_data)
        y_peak = np.zeros_like(x_data)
        y_peak[idx_peak] = cube[0]
        return y_spline + y_peak

    def likelihood(cube, idx_peak, x_data, y_data, x_knots, shift=3):
        y_model = model(cube, idx_peak, x_data, x_knots, shift=shift)
        lkl = skewnorm.logpdf(y_data - y_model,
                              cube[1],
                              loc=0,
                              scale=np.sqrt(np.abs(cube[2])))
        # Protect against inf
        lkl_inf = np.isinf(lkl)
        if np.sum(np.array(lkl_inf, dtype=int)):
            # This will give theoretically incorrect values
            # But only in cases to-be-rejected anyways
            lkl[lkl_inf] = np.min(lkl[~lkl_inf])
        return np.sum(lkl)

    def get_q_mu(mu_h, mu_d, cube, idx_peak, x_data, x_knots,
                 shift=3, tol=1e-4):
        # Generate data under assumption of mu - model + noise
        cube = np.concatenate([[mu_d], cube])
        y_model = model(cube, idx_peak, x_data, x_knots, shift=shift)
        y_noise = skewnorm.rvs(cube[1], loc=0, scale=np.sqrt(np.abs(cube[2])),
                               size=len(x_data))
        y_sim = y_model + y_noise
        del y_model, y_noise

        # Get L(mu_hat, theta_hat)
        bounds = [(None, None), (-10, 10), (0.1, 10)] + [(min(y_sim), max(y_sim)) for x in x_knots]
        popt = minimize(lambda x: -likelihood(x, idx_peak, x_data, y_sim, x_knots, shift=shift),
                        cube, method="L-BFGS-B", bounds=bounds, tol=tol)
        mu_hat = popt.x[0]
        if mu_hat > mu_h:
            return 0

        L_mu_hat_theta_hat = likelihood(popt.x, idx_peak, x_data, y_sim, x_knots, shift=shift)

        # Get L(mu, theta_hat_hat)
        def _likelihood(_params):  # negative log lkl for opt
            params = np.concatenate([[mu_h], _params])
            return -likelihood(params, idx_peak, x_data, y_sim, x_knots, shift=shift)

        popt = minimize(_likelihood, popt.x[1:], method="L-BFGS-B",
                        bounds=bounds[1:], tol=tol)
        L_mu_theta_hat_hat = -_likelihood(popt.x)

        # Return q_mu
        # print(-2*(L_mu_theta_hat_hat - L_mu_hat_theta_hat))
        return -2*(L_mu_theta_hat_hat - L_mu_hat_theta_hat)

    def get_mu_upper_quick(mu_d, cube, idx_peak, x_data, x_knots,
                           y_data, shift=3, tol=1e-8):
        def second_derivative(alpha, sigma, Lambda, theta):
            term1 = -2 * alpha**2 / (np.exp(alpha**2 * (Lambda - theta)**2 / sigma**2) * np.pi * sigma**2 * erfc(-alpha * (Lambda - theta) / (np.sqrt(2) * sigma))**2)
            term2 = -alpha**3 * np.sqrt(2 / np.pi) * (Lambda - theta) / (np.exp(alpha**2 * (Lambda - theta)**2 / (2 * sigma**2)) * sigma**3 * erfc(-alpha * (Lambda - theta) / (np.sqrt(2) * sigma)))
            return term1 + term2 - sigma**2

        # Generate data under mu_d assumption
        cube = np.concatenate([[mu_d], cube])
        y_model = model(cube, idx_peak, x_data, x_knots, shift=shift)
        y_noise = skewnorm.rvs(cube[1], loc=0, scale=np.sqrt(np.abs(cube[2])),
                               size=len(x_data))
        y_sim = y_model + y_noise
        del y_model, y_noise

        # Get mu_hat
        bounds = [(None, None), (-10, 10), (0.1, 10)] + [(min(y_sim), max(y_sim)) for x in x_knots]
        popt = minimize(lambda x: -likelihood(x, idx_peak, x_data, y_sim, x_knots, shift=shift),
                        cube, method="L-BFGS-B", bounds=bounds, tol=tol)
        mu_hat = popt.x[0]

        # Use likelihood to approx uncertainty on mu_hat
        c, d = np.polyfit(x_knots, popt.x[3:], deg=1)
        Lambda = y_data[idx_peak] - c*x_data[idx_peak] - d
        Fisher = second_derivative(popt.x[1], popt.x[2], Lambda, mu_hat)

        # Calculate 95% CL upper limit:
        return y_sim[fi], mu_hat + np.sqrt(-1./Fisher) * 1.64, np.sqrt(-1./Fisher)

    shift = 3
    # For each frequency:
    bkg, mu_upper, sigma = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
    for fi in tqdm(np.random.choice(np.arange(len(X)), size=len(X)//pruning),
                   desc="inner", position=1, leave=False):
    # for fi in trange(0, len(X), pruning, desc="inner", position=1, leave=False):
        # 0) Background model given by best_fit from spline opti
        # 0.1) Initialise variables
        # initial_guess = np.zeros(len(best_fit) + 1)
        # initial_guess[1:] = best_fit
        # bounds = [(None, None), (-10, 10), (0.1, 10)] + [(min(Y), max(Y)) for x in x_knots]

        # TMP for Friday meeting
        # Use Likelihood approx to get mu_upper without MC
        mu_data = 0
        bkg[fi], mu_upper[fi], sigma[fi] = get_mu_upper_quick(
            mu_data, best_fit, fi, X, x_knots, Y, shift=shift, tol=1e-8)
    return bkg, mu_upper, sigma

        # 1) Choose mu_initial != 0
        # mu = 1
        # 2) get q(mu|0) and q(mu|mu) M times
        # M = 1000
        # f_q_mu_0, f_q_mu_mu = np.zeros(M), np.zeros(M)
        # for i in trange(M, desc="f_q_mu"):
        #     f_q_mu_0[i] = get_q_mu(mu, 0, best_fit, fi, X, x_knots,
        #                            shift=shift, tol=1e-8)
        #     f_q_mu_mu[i] = get_q_mu(mu, mu, best_fit, fi, X, x_knots,
        #                             shift=shift, tol=1e-8)
        # 3.0) Plot (tmp)
        # bins = np.linspace(min(min(f_q_mu_mu), min(f_q_mu_0)),
        #                    max(max(f_q_mu_mu), max(f_q_mu_0)), 50)
        # mask = f_q_mu_0 == 0
        # f_q_mu_0 = f_q_mu_0[~mask]
        # test = test[~mask]

        # ax = hist.plot_hist(f_q_mu_0, bins, logy=True, label=r"$f(q_{\mu}|0)$")
        # # ax = hist.plot_hist(f_q_mu_mu, bins, ax=ax, color="C1", label=r"$f(q_{\mu}|\mu)$")
        # # ax.axvline(np.median(f_q_mu_0), color="r", linewidth=2, linestyle="--")
        # ax.legend(loc="upper right")

        # hist.plot_hist(test, np.linspace(min(test), max(test), 30))
        # print(np.mean(f_q_mu_0), np.std(f_q_mu_0, ddof=1))
        # print(np.mean(test), np.std(test, ddof=1))

        # # 3) Get p-value
        # p_mu = np.sum(f_q_mu_mu[f_q_mu_mu > np.median(f_q_mu_0)]) / np.sum(f_q_mu_mu)
        # print(f"p_mu: {p_mu:.2e}")
        # p_mu = np.sum(np.array([f_q_mu_mu > np.median(f_q_mu_0)], dtype=int)) / M
        # print(f"p_mu: {p_mu:.2e}")
        # plt.show()

        # 4) Repeat 3,4 N times, then look at mu_upper distribution
        # This will not be feasible unless I find at least 10^3 speedup
    pass


def main():
    """Calculate experimental sensitivity based on simulated data."""
    # Analysis variables
    segment_size = 10000

    # Get "data"
    fmin = 10  # Hz
    fmax = 8192  # Hz
    resolution = 1e-6

    Jdes = utils.Jdes(fmin, fmax, resolution)
    X = np.log(np.logspace(np.log10(fmin), np.log10(fmax), Jdes))

    # Get model parameters
    x_knots = np.array([2.30258509, 3.04941017, 3.79623525,
                        8.65059825, 8.95399594, 8.97733423, 9.02401079])
    model_params = np.load("bkg_model_params.npy")
    model = models.model_spline(model_params, x_knots, shift=0)(X)

    # Generate noise
    print("Generate noise..")
    fit_data = np.load("fit_results_skew.npz", allow_pickle=True)
    idx, interp = fit_data["idx"], fit_data["interp"]
    noise = np.zeros_like(X)
    for i, (start, end) in enumerate(idx):
        mu, sigma, alpha = list(map(lambda f: f(i), interp))
        noise[start:end] = skewnorm.rvs(alpha, loc=mu, scale=sigma, size=end-start)
        if i == 0:
            first_start = start
        elif i == len(idx) - 1:
            last_end = end

    X, noise = X[first_start:last_end], noise[first_start:last_end]
    Y = noise + model[first_start:last_end]

    start = 0
    pruning = 10000
    bkg, upper_limits, sigma = [np.zeros(len(X)) for x in range(3)]
    for end in trange(segment_size, len(X)+segment_size, segment_size,
                      desc="outer", position=0):
        bkg[start:end], upper_limits[start:end], sigma[start:end] =\
            get_upper_limits(X[start:end], Y[start:end], pruning)

        # Loop condition
        start = end

    mask = bkg == 0
    bkg, upper_limits, sigma = bkg[~mask], upper_limits[~mask], sigma[~mask]
    x = np.exp(X[~mask])

    # Prelim plot
    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Gaussian approximation")
    ax.plot(np.exp(X), np.exp(Y), label="Noise level", zorder=1)
    ax.errorbar(x, np.exp(bkg + upper_limits),
                np.exp(bkg + upper_limits)*sigma,
                fmt=".", label="Upper limit", zorder=1)

    # TMP
    def running_average(data, window_size):
        half_window = window_size // 2
        return np.array([np.mean(data[max(0, i - half_window) : min(len(data), i + half_window + 1)])
                        for i in range(len(data))])

    window = 50
    bkg_smooth = running_average(bkg, window)
    upper_lim_raw_smooth = running_average(upper_limits, window)
    upper_lim_smooth = np.exp(bkg_smooth + upper_lim_raw_smooth)

    sigma_smooth_raw = running_average(sigma, window)
    sigma_smooth = upper_lim_smooth * sigma_smooth_raw

    ax.plot(x, upper_lim_smooth, color="C1", zorder=2)
    ax.fill_between(x, upper_lim_smooth+sigma_smooth, upper_lim_smooth-sigma_smooth,
                    color="C1", alpha=.3, zorder=3, label="Upper limit")
    ax.legend(loc="best")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")

    df = pd.DataFrame({"upper_limit": upper_lim_smooth,
                       "uncertainty": sigma_smooth,
                       "frequency": x})
    print(df)    
    df.to_csv("preliminary_epsilon10_1243393026_1243509654_H1_upper_limits.csv")
    # TMP

    plt.figure()
    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_title("mu")
    ax.errorbar(x, upper_limits, sigma, color="C1", fmt=".")

    plt.figure()
    ax = plt.subplot(111)
    ax.set_title("bkg")
    ax.set_xscale("log")
    ax.plot(x, bkg, color="C1")
    plt.show()


main()
