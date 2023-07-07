"""
Calculate experimental upper limit based on simulated data.

Compare MC-based method with approximations.
"""
import warnings
from tqdm import trange, tqdm
import numpy as np
import pandas as pd
from multiprocessing import Pool
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize, curve_fit
from scipy.stats import skewnorm, norm
from scipy.special import erf, erfc
from scipy.integrate import quad
from scipy import constants
import ultranest
# Project imports
import models
import utils
import hist


# FIXME: TMP
# warnings.filterwarnings("ignore")


def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


def gradient_descent(f, x0, learning_rate, max_iterations, threshold, bounds):
    x = x0

    for i in range(max_iterations):
        gradient = numerical_derivative(f, x)

        x = x - learning_rate * gradient
        x = max(min(x, bounds[1]), bounds[0])

        if abs(gradient) < threshold:
            break

    return x


def get_efficiency_var(k, M):
    """
    Get Bayesian efficiency variance.
    
    see https://facultystaff.richmond.edu/~ggilfoyl/research/EfficiencyErrors.pdf.
    """
    return (k + 1) * (k + 2) / float((M + 2) * (M+3)) - ((k+1) / (M+2))**2


def approx_logskew(x, alpha):
    _sgn = np.sign(alpha)
    alpha = np.abs(alpha)
    x = _sgn * x.copy()

    output = np.zeros_like(x)
    one_over_alpha = 1. / alpha
    prefix = 1. / np.sqrt(2*np.pi)

    m1 = (x >= -3.*one_over_alpha) & (x < -one_over_alpha)
    m2 = (x >= -one_over_alpha) & (x < one_over_alpha)
    m3 = (x >= one_over_alpha) & (x < 3*one_over_alpha)
    m4 = x >= 3*one_over_alpha

    _x = x[m1]
    output[m1] = 0.125 * prefix * np.exp(-0.5*_x**2) * (9*alpha*_x + 3*alpha**2*_x**2 + alpha**3*_x**3/3. + 9)
    _x = x[m2]
    output[m2] = 0.25 * prefix * np.exp(-0.5*_x**2) * (3*alpha*_x - alpha**3*_x**3/3. + 4)
    _x = x[m3]
    output[m3] = 0.125 * prefix * np.exp(-0.5*_x**2) * (9*alpha*_x - 3.*alpha**2*_x**2 + alpha**3*_x**3/3. + 7)
    _x = x[m4]
    output[m4] = 2. * prefix * np.exp(-0.5*_x**2)

    return np.log(output)


def asymptotic(x, mu, mu_prime, sigma):
    x = np.atleast_1d(x)
    #x[x == 0] = norm.cdf((mu - mu_prime) / sigma)
    norm = 0.5*(1 + erf((mu - mu_prime) / (np.sqrt(2)*sigma)))
    f_x = 0.5 / np.sqrt(2 * np.pi * x) * np.exp(-0.5 * (np.sqrt(x) - (mu - mu_prime) / sigma)**2)
    return f_x / norm


def run_simple_spline_fit(X, Y, x_knots, verbose=False, nlive=64):
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
    popt = minimize(lambda x: -_likelihood(x), initial_guess, method="Powell",
                    bounds=bounds, options={"ftol": 1e-4})
    initial_guess = popt.x

    # Run UltraNest
    sampler = ultranest.ReactiveNestedSampler(
        parameters, _likelihood, transform=_prior,
        vectorized=False, resume="overwrite"
    )
    results = sampler.run(viz_callback=False,
                          min_num_live_points=nlive)

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


def get_q_mu_for_index(args):
    # mu, mu_0_or_mu, best_fit, idx_peak, X, x_knots, shift, M, i = args
    # return get_q_mu(mu, mu_0_or_mu, best_fit, idx_peak, X, x_knots, shift=shift)
    return simultaneous_q_mu(*args[:6])


def simultaneous_q_mu(mu_h, cube, idx_peak, x_data, x_knots,
                      shift=3, tol=1e-10, pruning=10):
    # Generate data
    cube = np.concatenate([[0], cube])
    y_model = model(cube, idx_peak, x_data, x_knots, shift=shift)
    y_noise = skewnorm.rvs(cube[1], loc=0, scale=np.sqrt(np.abs(cube[2])),
                           size=len(x_data))
    y_data = y_model + y_noise
    del y_model, y_noise

    peak_val = y_data[idx_peak]
    x_pos = x_data[idx_peak]
    x_data = np.delete(x_data, idx_peak)
    y_data = np.delete(y_data, idx_peak)
    x_pruned, y_pruned = x_data[::pruning], y_data[::pruning]

    # Optimize over hunk
    bounds = [(None, None), (0.01, 10)] + [(min(y_data), max(y_data)) for x in x_knots]
    def _lkl_0(params):
        _params = np.concatenate([[0], params])
        y_model = model(_params, 0, x_pruned, x_knots, shift=shift)
        lkl = skewnorm.logpdf(y_pruned - y_model,
                              _params[1],
                              scale=np.sqrt(_params[2]))
        # Protect against inf
        lkl_inf = np.isinf(lkl)
        n_inf = np.sum(np.array(lkl_inf), dtype=int)
        if n_inf == len(lkl):
            return -1e4 * len(lkl)  # empirical
        if n_inf:
            lkl[lkl_inf] = np.min(lkl[~lkl_inf])
        return np.sum(lkl)

    popt_hunk = minimize(lambda x: -_lkl_0(x),
                         cube[1:], method="L-BFGS-B", bounds=bounds,
                         options={"ftol": tol})
    y_hunk = model(np.concatenate([[0], popt_hunk.x]),
                   0, x_data, x_knots, shift=shift)
    lkl_hunk = np.sum(skewnorm.logpdf(y_data - y_hunk,
                                      popt_hunk.x[0],
                                      scale=np.sqrt(popt_hunk.x[1])))

    # Now optimize over just the peak, for mu=mu and mu=0
    popt_peak_zero = minimize(lambda x: -skewnorm.logpdf(
        peak_val - model(
            np.concatenate([x, popt_hunk.x]), 0, [x_pos], x_knots, shift=shift),
        cube[1], scale=np.sqrt(np.abs(cube[2]))),
                            mu_h, tol=tol)
    popt_peak_mu = minimize(lambda x: -skewnorm.logpdf(
        peak_val + mu_h - model(
            np.concatenate([x, popt_hunk.x]), 0, [x_pos], x_knots, shift=shift),
        cube[1], scale=np.sqrt(np.abs(cube[2]))),
                            mu_h, tol=tol)

    # Maximum likelihood given that zero or mu_h are in the data
    zero_L_mu_hat_theta_hat = lkl_hunk - popt_peak_zero.fun
    mu_L_mu_hat_theta_hat = lkl_hunk - popt_peak_mu.fun

    # Now get the value of L in the peak for different mu
    # 1. Data has zero, model assumes mu_h
    # 2. Data has mu_h, model assumes mu_h
    y_model_mu_h = model(np.concatenate([[mu_h], popt_hunk.x]),
                         0, [x_pos], x_knots, shift=shift)
    L_mu_zero = skewnorm.logpdf(peak_val - y_model_mu_h,
                                cube[1], scale=np.sqrt(np.abs(cube[2])))[0]
    L_mu_h = skewnorm.logpdf(peak_val + mu_h - y_model_mu_h,
                                cube[1], scale=np.sqrt(np.abs(cube[2])))[0]

    zero_L_mu_theta_hat_hat = lkl_hunk + L_mu_zero
    mu_L_mu_theta_hat_hat = lkl_hunk + L_mu_h

    # Return q_mu|0, q_mu|mu
    return -2*(zero_L_mu_theta_hat_hat - zero_L_mu_hat_theta_hat),\
        -2*(mu_L_mu_theta_hat_hat - mu_L_mu_hat_theta_hat)


def get_q_mu(mu_h, mu_d, cube, idx_peak, x_data, x_knots, shift=3, tol=1e-11):
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
                    cube, method="L-BFGS-B", bounds=bounds,
                    options={"ftol": tol})
    mu_hat = popt.x[0]
    if mu_hat > mu_h:
        return 0

    L_mu_hat_theta_hat = likelihood(popt.x, idx_peak, x_data, y_sim, x_knots, shift=shift)

    # Get L(mu, theta_hat_hat)
    def _likelihood(_params):  # negative log lkl for opt
        params = np.concatenate([[mu_h], _params])
        return -likelihood(params, idx_peak, x_data, y_sim, x_knots, shift=shift)

    popt = minimize(_likelihood, popt.x[1:], method="Powell",
                    bounds=bounds[1:],
                    options={"ftol": tol})
    L_mu_theta_hat_hat = -_likelihood(popt.x)

    # Return q_mu
    return -2*(L_mu_theta_hat_hat - L_mu_hat_theta_hat)

def model(cube, idx_peak, x_data, x_knots, shift=3):
    # Spline background + single delta peak model
    y_knots = cube[shift:len(x_knots)+shift]
    y_model = CubicSpline(x_knots, y_knots, extrapolate=False)(x_data)
    y_model[idx_peak] += cube[0]
    return y_model

def likelihood(cube, idx_peak, x_data, y_data, x_knots, shift=3):
    y_model = model(cube, idx_peak, x_data, x_knots, shift=shift)
    # lkl = approx_logskew((y_data - y_model)/scale, cube[1]) / scale
    lkl = skewnorm.logpdf(y_data - y_model,
                          cube[1],
                          loc=0,
                          scale=np.sqrt(np.abs(cube[2])))
    # Protect against inf
    lkl_inf = np.isinf(lkl)
    if np.sum(np.array(lkl_inf, dtype=int)):
        # This will give theoretically incorrect values
        # But only in cases to-be-rejected anyways
        # print([f"{x:.1e}" for x in cube])
        lkl[lkl_inf] = np.min(lkl[~lkl_inf])
    return np.sum(lkl)


def second_derivative(alpha, sigma, Lambda, theta, verbose=False):
    """Calculate the second derivative the skew normal of (y-cubic(x)-theta)/sigma w.r.t. theta."""
    term1 = -2 * alpha**2 / (np.exp(alpha**2 * (Lambda - theta)**2 / sigma**2) * np.pi * sigma**2 * erfc(-alpha * (Lambda - theta) / (np.sqrt(2) * sigma))**2)
    term2 = -alpha**3 * np.sqrt(2 / np.pi) * (Lambda - theta) / (np.exp(alpha**2 * (Lambda - theta)**2 / (2 * sigma**2)) * sigma**3 * erfc(-alpha * (Lambda - theta) / (np.sqrt(2) * sigma)))
    if verbose:
        print("Terms: ", term1, term2, -sigma**2)
    return term1 + term2 - sigma**-2


def get_upper_limits(X, Y, idx_peak):
    """Use Cowan formulae and MC methods to get upper limits."""
    def get_mu_upper_quick(cube, idx_peak, x_data, x_knots,
                           y_data, shift=3, tol=1e-10):
        # 2. Calculate using dlnL/dmu from data
        bounds = [(None, None), (-10, 10), (-10, 10)] +\
                [(min(y_data), max(y_data)) for x in x_knots]
        popt = minimize(lambda x: -likelihood(x, idx_peak, x_data, y_data, x_knots, shift=shift),
                        np.concatenate([[0], cube]),
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={"ftol": tol, "gtol": tol})
        # TODO: see if using q_mu_tilda makes a difference?
        # Build spline to get Lambda
        y_knots = popt.x[shift:len(x_knots)+shift]
        y_spline = CubicSpline(x_knots, y_knots, extrapolate=False)
        Lambda = y_data[idx_peak] - y_spline(x_data[idx_peak])
        # No need to sum because of the delta function
        Fisher = second_derivative(popt.x[1], np.sqrt(popt.x[2]),
                                   Lambda, popt.x[0])

        # Return mu_asimov, mu_lnL, sigma
        # FIXME Abandonning mu_asimov for now
        return 0, popt.x[0], np.sqrt(-1./Fisher)

    # 0) Optimise spline for background model
    # TODO
    # x_knots, best_fit = optimise_x_knots(X[::pruning], Y[::pruning], verbose=True)
    # 0.1) Fit background-only model to data for later generation
    shift = 3
    n_knots = 2  # VERY TMP
    x_knots = np.linspace(min(X), max(X), n_knots)
    a, b = np.polyfit(X, Y, deg=1)
    y_knots = a*x_knots + b  # VERY TMP

    def _likelihood_no_peak(x):
        params = np.concatenate([[0], x])
        return -likelihood(params, 0, X, Y, x_knots, shift=shift)

    bounds = [(-10, 10), (.001, 10)] + [(min(Y), max(Y)) for x in x_knots]
    initial_guess = np.concatenate([[0, 0.3], y_knots])
    popt = minimize(_likelihood_no_peak, initial_guess,
                    method="Powell", options={"ftol": 1e-8}, bounds=bounds)
    best_fit = popt.x  # Does not contain mu

    # 1) Use Likelihood approx to get mu_upper without MC
    mu_asimov, mu_lnL, sigma = get_mu_upper_quick(
        best_fit, idx_peak, X, x_knots, Y, shift=shift, tol=1e-15)

    # 2) Now get answer through full MC calculation
    # TODO: also try this one with q_mu_tilda?
    # 2.0) Initialiase analysis variables
    M = 2000
    alpha = 0.05  # 95% C.L.
    threshold, max_iterations = 0.01, 100

    def get_p_value(mu, M, num_workers=10, return_p=True, verbose=False):
        # return fit_p_value(mu, 1, 1) + np.random.normal(size=1, scale=0.2)[0], 1
        # Arguments for get_q_mu_for_index
        # args_0 = [(mu, 0, best_fit, idx_peak, X, x_knots, shift, M, i) for i in range(M)]
        # args_mu = [(mu, mu, best_fit, idx_peak, X, x_knots, shift, M, i) for i in range(M)]
        args = [(mu, best_fit, idx_peak, X, x_knots, shift, M, i) for i in range(M)]

        # Create a pool of worker processes
        with Pool(num_workers) as pool:
            # Use imap instead of map for lazy evaluation
            # f_q_mu_0 = np.array(list(tqdm(pool.imap(get_q_mu_for_index, args_0),
            #                               desc="f_q_mu_0", position=2, leave=False, total=M)))
            # f_q_mu_mu = np.array(list(tqdm(pool.imap(get_q_mu_for_index, args_mu),
            #                                desc="f_q_mu_mu", position=2, leave=False, total=M)))

            results = list(tqdm(pool.imap(get_q_mu_for_index, args),
                                desc="f_q_mu", position=2, leave=False, total=M))
            f_q_mu_0 = np.array([x[0] for x in results])
            f_q_mu_mu = np.array([x[1] for x in results])

        f_q_mu_0 = f_q_mu_0[~np.isnan(f_q_mu_0)]
        f_q_mu_mu = f_q_mu_mu[~np.isnan(f_q_mu_mu)]
        if verbose:
            bins = np.linspace(min(f_q_mu_mu), max(f_q_mu_0), 100)
            ax = hist.plot_hist(f_q_mu_0, bins, logy=True, color="C0",
                                label=r"$f(q_{\mu}|0)$", density=True)
            ax = hist.plot_hist(f_q_mu_mu, bins, ax=ax, color="C1",
                                label=r"$f(q_{\mu}|\mu)$", density=True)
            bin_centers = (bins[1:] + bins[:-1]) / 2.
            ax.plot(bin_centers, asymptotic(bin_centers, mu, 0, sigma), color="C0")
            ax.plot(bin_centers, asymptotic(bin_centers, mu, mu, sigma), color="C1")
            ax.axvline(np.median(f_q_mu_0), color="r", label="Median", linestyle="--")
            ax.legend(loc="best")
            n = np.sum(np.array([f_q_mu_mu > np.median(f_q_mu_0)], dtype=int))
            print("p-values ingredients:", n, M)
            # Calculate median analytically
            _med = minimize(lambda x: (quad(lambda t: asymptotic(t, mu, 0, sigma), 0, x)[0]
                                       - 0.5)**2,
                            1.5, bounds=[(1e-10, None)], method="Powell",
                            options={"ftol": 1e-2}).x
            ax.axvline(_med, color="C0", label="median", linestyle="--")
            plt.show()
        # Compute and return the p-value
        if return_p:
            n = np.sum(np.array([f_q_mu_mu > np.median(f_q_mu_0)], dtype=int))
            var = np.sqrt(get_efficiency_var(n, M))
            return n / float(M), np.sqrt(var)
        return f_q_mu_0, f_q_mu_mu

    # 2.1) Get f_q_mus for an example mu and compare to formulae
    # 2.2) Optimize
    def fit_p_value(mu, sigma, a):
        """Analytically calculated Expected value of f_q_mu_0 used to integrate f_q_mu_mu."""
        term1 = 1/(np.sqrt(2*np.pi)*sigma)*mu*np.exp(-0.5*(mu/sigma)**2)
        term2 = 0.5*(mu**2 + sigma**2) / sigma**2 * (1 + erf(mu / (np.sqrt(2)*sigma)))
        return a*erfc(np.sqrt((term1 + term2) / 2.))

    mus = [3, 2, .1]
    ps = [get_p_value(x, M) for x in mus]
    # ps = [(0.0, 0.0009970084765641995), (0.0, 0.0009970084765641995), (0.01, 0.0032901445728277137)]
    print(f"SIGMA: {sigma:.2f}")
    print(mus)
    print(ps)
    for _ in trange(max_iterations, position=1, leave=False, desc="MC p-value"):
        # Fit to guess new mu
        def _lkl(sigma, a):
            residuals = np.array([x[0] for x in ps]) - fit_p_value(np.array(mus), sigma, a)
            return np.sum(norm.logpdf(residuals, scale=np.array([x[1] for x in ps])))
        popt = minimize(lambda x: -_lkl(*x), np.array([.43, 1]),
                        method="Powell",
                        bounds=[(0, None), (0, None)],
                        options={"ftol": 1e-15})
        print(popt)
        plt.figure()
        plt.errorbar(mus,
                     np.array([x[0] for x in ps]),
                     np.array([x[1] for x in ps]),
                     fmt=".")
        plt.axhline(alpha, linestyle="--", color="r", linewidth=2.)
        x = np.linspace(min(mus)*0.9, max(mus)*1.1, 100)
        plt.plot(x, fit_p_value(x, .43, 1), label="Initial guess")
        plt.plot(x, fit_p_value(x, *popt.x), label="Fit")
        plt.xlabel(r"$\mu$ (hypothesis)")
        plt.ylabel("p-value")
        plt.legend(loc="upper right")
        plt.show()

        # Now that data is fit well, use popt to estimate mu that will give alpha
        min_popt = minimize(lambda x: (fit_p_value(x, *popt.x) - alpha)**2,
                            1, bounds=[(0, None)], method="Powell",
                            options={"ftol": 1e-10})
        # min_popt = gradient_descent(lambda x: np.abs(fit_p_value(x, *popt.x) - alpha), x0, learning_rate, max_iterations, threshold, bounds
        print(min_popt)
        mus += [min_popt.x[0]]
        if mus[-1] <= 0:
            M *= 2
            mus = list(np.delete(mus, -1))
            # Replace lowest p-value with updated one
            lowest_mu_idx = np.argmin(mus)
            tqdm.write("Fit supports negative mu, re-do lowest point")
            ps[lowest_mu_idx] = get_p_value(mus[lowest_mu_idx], M, verbose=True)
            continue
        ps += [get_p_value(mus[-1], M, verbose=True)]
        tqdm.write(f"mu_guess: {mus[-1]:.1e}, p_new: {ps[-1][0]:.2} +- {ps[-1][1]:.1e}")

        # Check if the new p-value is close enough to alpha
        if abs(ps[-1][0] - alpha) < threshold:
            print(f"Found mu: {mus[-1]:.1e} with p-value: {ps[-1][0]:.2} +- {ps[-1][1]:.1e}")
            break
    else:
        print("Did not converge to a solution.")
    mu_MC = mus[-1]

    # f_q_mu_0, f_q_mu_mu = get_p_value(mu_MC, 3000, 11, False)
    # # 4) Plot hists with formulae for confirmation
    # bins = np.linspace(0, max(f_q_mu_0), 100)
    # ax = hist.plot_hist(f_q_mu_0, bins, logy=True, label=r"$f(q_{\mu}|0)$", density=True)
    # ax = hist.plot_hist(f_q_mu_mu, bins, ax=ax, color="C1", label=r"$f(q_{\mu}|\mu)$", density=True)

    # bin_centers = (bins[1:] + bins[:-1]) / 2.
    # ax.plot(bin_centers, asymptotic(bin_centers, mu_MC, 0, sigma), color="C0")
    # ax.plot(bin_centers, asymptotic(bin_centers, mu_MC, mu_MC, sigma))
    # ax.axvline(np.median(f_q_mu_0), color="r", linewidth=2, linestyle="--")
    # ax.legend(loc="upper right")
    # plt.show()

    tqdm.write(f"-- fi: {idx_peak} --")
    tqdm.write(f"mu_asimov:\t{mu_asimov:.2f}\nmu_lnL:\t{mu_lnL:.2f}\nmu_MC:\t{mu_MC:.2f}")
    tqdm.write(f"sigma: {sigma:.2f}, bkg: {Y[idx_peak]:.2f}")
    return mu_asimov, mu_lnL, sigma, mu_MC, Y[idx_peak]


def get_upper_limits_approx(X, Y, idx_peak):
    """Use asymptotic approximatinos to get upper limits (with uncertainty)."""
    # Get starting estimate of background-only fit
    # x_knots, best_fit = optimise_x_knots(X[::pruning], Y[::pruning], verbose=True)
    # TODO: only need to do this once per segment
    # Meanwhile tmp:
    shift = 3
    n_knots = 2  # VERY TMP
    x_knots = np.linspace(min(X), max(X), n_knots)
    a, b = np.polyfit(X, Y, deg=1)
    y_knots = a*x_knots + b
    # def _likelihood_no_peak(x):
    #     params = np.concatenate([[0], x])
    #     return -likelihood(params, 0, X, Y, x_knots, shift=shift)

    # bounds = [(-10, 10), (.01, 10)] + [(min(Y), max(Y)) for x in x_knots]
    # initial_guess = np.concatenate([[0, 0.3], y_knots])
    # popt = minimize(_likelihood_no_peak, initial_guess,
    #                 method="Powell", options={"ftol": 1e-10}, bounds=bounds)
    # best_fit = popt.x  # Does not contain mu
    best_fit = np.concatenate([[0, 0.3], y_knots])
    ### END TMP ###

    # Perform maximum likelihood fit to get mu_hat, theta_hat
    bounds = [(None, None), (None, None), (.01, None)]  # mu, alpha, sigma
    bounds += [(min(Y), max(Y)) for y in y_knots]
    popt = minimize(lambda x: -likelihood(x, idx_peak, X, Y, x_knots, shift=shift),
                    x0=np.concatenate([[0], best_fit]),
                    method="L-BFGS-B",
                    options={"ftol": 1e-12, "gtol": 1e-20},
                    bounds=bounds)

    # Now get sigma_mu_hat from dlnL/dmu
    y_knots = popt.x[shift:len(x_knots)+shift]
    y_spline = CubicSpline(x_knots, y_knots, extrapolate=False)
    Lambda = Y[idx_peak] - y_spline(X[idx_peak])
    Fisher = second_derivative(popt.x[1], np.sqrt(np.abs(popt.x[2])),
                               Lambda, popt.x[0], verbose=False)

    # Return mu_hat, sigma_dlnL, bkg
    bkg = y_spline(X[idx_peak])
    return popt.x[0], np.sqrt(-1. / Fisher), bkg


def process_segment(args):
    """Calculate an upper limit - wrapper for multiprocessing."""
    i, x_subset, y_subset, fi, lo = args
    return i, get_upper_limits_approx(x_subset, y_subset, fi - lo)


def smooth_curve(y, w):
    """Smooth data y with length w."""
    output = np.zeros_like(y)
    for i in trange(len(y)):
        subset = y[max(0, i-w):i+w]
        output[i] = np.median(subset[~np.isnan(subset)])
    return output


def main():
    """Get the upper limit in every (nth?) bin using asymptotic approximations."""
    # Analysis variables
    segment_size = 10000
    fmin = 10  # Hz
    fmax = 8192  # Hz
    resolution = 1e-6

    # Process variables
    Jdes = utils.Jdes(fmin, fmax, resolution)
    X = np.log(np.logspace(np.log10(fmin), np.log10(fmax), Jdes))

    # Get model parameters
    x_knots = np.array([2.30258509, 3.04941017, 3.79623525,
                        8.65059825, 8.95399594, 8.97733423, 9.02401079])
    model_params = np.load("bkg_model_params.npy")
    y_model = models.model_spline(model_params, x_knots, shift=0)(X)

    # Generate noise
    print("Generate noise..")
    fit_data = np.load("fit_results_skew.npz", allow_pickle=True)
    idx, interp = fit_data["idx"], fit_data["interp"]
    y_noise = np.zeros_like(X)
    _params = np.zeros((len(idx), 2))
    for i, (start, end) in enumerate(idx):
        mu, sigma, alpha = list(map(lambda f: f(i), interp))
        y_noise[start:end] = skewnorm.rvs(alpha, loc=mu, scale=sigma, size=end-start)
        _params[i, :] = alpha, sigma
        if i == 0:
            first_start = start
        elif i == len(idx) - 1:
            last_end = end

    X, y_noise = X[first_start:last_end], y_noise[first_start:last_end]
    Y = y_noise + y_model[first_start:last_end]

    # Calculate upper limits
    pruning = 1000
    freq_indices = np.arange(segment_size//2, len(X)-segment_size//2, pruning)
    # upper_limit_data = np.zeros((3, len(freq_indices)))
    # num_processes = 11
    # with Pool(num_processes) as p:
    #     # Prepare the data for each worker with only subsets of X and Y
    #     args = []
    #     for i, fi in enumerate(freq_indices):
    #         lo, hi = fi - segment_size // 2, fi + segment_size // 2
    #         x_subset = X[lo:hi].copy()
    #         y_subset = Y[lo:hi].copy()
    #         args.append((i, x_subset, y_subset, fi, lo))

    #     # Process the data in parallel
    #     results = list(tqdm(p.imap(process_segment, args), total=len(args), position=0))

    # # Update the results back in the upper_limit_data array
    # for i, upper_limit in results:
    #     upper_limit_data[:, i] = upper_limit

    # # # Checkpoint step
    # np.save("upper_limit_data_seg10k_pru1k.npy", upper_limit_data)
    # upper_limit_data = np.load("upper_limit_data.npy")
    upper_limit_data = np.load("upper_limit_data_seg10k_pru1k.npy")

    # Save to df
    x, y = np.exp(X), np.exp(Y)
    upper_limit = np.exp(upper_limit_data[2, :]
                         + upper_limit_data[0, :]
                         + 1.64*upper_limit_data[1, :])
    sigma_lnL = upper_limit * upper_limit_data[1, :]

    window = 50
    upper_limit_smooth = smooth_curve(upper_limit, window)
    sigma_lnL_smooth = smooth_curve(sigma_lnL, window)
    background_smooth = smooth_curve(upper_limit_data[2, :], window)

    df = pd.DataFrame({"frequency": x[freq_indices],
                       "upper_limit": upper_limit,
                       "uncertainty": sigma_lnL,
                       "upper_limit_smooth": upper_limit_smooth,
                       "uncertainty_smooth": sigma_lnL_smooth,
                       "background_spline": background_smooth})
    df.to_csv("preliminary_epsilon10_1243393026_1243509654_H1_upper_limits.csv")

    # Plot results
    plt.figure()
    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(x, y, label="Simulated data", zorder=1)
    ax.fill_between(x[freq_indices],
                    upper_limit_smooth + sigma_lnL_smooth,
                    upper_limit_smooth - sigma_lnL_smooth,
                    color="C1", alpha=.33, zorder=2)
    ax.plot(x[freq_indices], upper_limit_smooth, color="C1",
            label="Upper limit (95% C.L.)", zorder=2)

    # Nice things
    ax.legend(loc="best")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    plt.show()


def test():
    df = pd.read_csv("preliminary_epsilon10_1243393026_1243509654_H1_upper_limits.csv")
    calib = np.loadtxt("../shared_git_data/Calibration_factor_A_star.txt", delimiter="\t")

    f_calib = interp1d(calib[:, 1], calib[:, 0])
    m = (df["frequency"] >= min(calib[:, 1])) & (df["frequency"] <= max(calib[:, 1]))
    freq = df["frequency"][m]
    upper_lim = df["upper_limit_smooth"][m]
    sigma = df["uncertainty_smooth"][m]
    bkg = np.exp(df["background_spline"][m])
    a_star = f_calib(freq)

    n, l, rho_local = 1, 6e-3, 0.3  # GeV/cm3
    mass_eV = freq * constants.h / constants.e
    beta = (n*l*constants.hbar / (mass_eV * constants.c * a_star))**2 *\
        2 * rho_local / (freq*2*np.pi * 1e-6)
    coupling = np.sqrt(beta * freq * 1e-6 / (upper_lim - bkg))

    # Geo limits for comparison
    geo_data = np.loadtxt("geo_limits.csv", delimiter=",")

    # Plot all
    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(freq, coupling, label="LIGO", linewidth=2., color="C1")
    ax.plot(geo_data[:, 0], geo_data[:, 1], label="GEO600", linewidth=2, color="C0")
    ax.fill_between(freq, coupling + 0.5*sigma*coupling/(upper_lim - bkg),
                    coupling - 0.5*sigma*coupling/(upper_lim - bkg),
                    color="C1", alpha=.33)

    # Add mass axis & nice things
    ticks = np.array([1e-13, 1e-12, 1e-11])
    axTop = ax.twiny()
    axTop.set_xscale("log")
    axTop.set_xlim(ax.get_xlim())
    axTop.set_xticks(ticks * constants.e / constants.h)
    axTop.set_xticklabels(ticks)
    ax.set_xlabel("Frequency (Hz)")
    axTop.set_xlabel("DM mass (eV)")
    ax.set_ylabel(r"$\Lambda_i$ (GeV)")
    ax.set_title("PRELIMINARY 95% UPPER LIMITS")
    ax.legend(loc="upper left")
    print(ax.get_xlim())
    print(ax.get_ylim())
    ax.text(np.exp(0.02*(np.log(ax.get_xlim()[1]) - np.log(ax.get_xlim()[0])) + np.log(ax.get_xlim()[0])),
            np.exp(0.35*(np.log(ax.get_ylim()[1]) - np.log(ax.get_ylim()[0])) + np.log(ax.get_ylim()[0])),
            "PRELIMINARY", fontsize=50, color="grey", alpha=.5)
    plt.show()


if __name__ == '__main__':
    # main()
    test()
