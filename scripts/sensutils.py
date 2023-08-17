"""Collect useful common functions for the sensitivity analysis."""
import os
import json
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import erf
from scipy.stats import norm, skew
from scipy.signal import convolve


def get_skewnorm_p0(data):
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

    return alpha, mu, sigma

def get_mode_skew(mu, sigma, alpha):
    """Return (wikipedia) numerical approximation of mode."""
    delta = alpha / np.sqrt(1 + alpha**2)
    mu_z = np.sqrt(2 / np.pi) * delta
    sigma_z = np.sqrt(1 - mu_z**2)
    gamma_1 = (4 - np.pi) / 2. * mu_z**3 / (1 - mu_z**2)**1.5
    m0 = mu_z - gamma_1*sigma_z/2. - np.sign(alpha)/2.*np.exp(-2*np.pi/np.abs(alpha))
    return mu + sigma*m0


def logpdf_skewnorm(x, alpha, loc=0, scale=1):
    """Implement skewnorm.logpdf for speed."""
    norm_term = -0.5*((x - loc) / scale)**2
    skew_term = np.log(0.5*(1. + erf(alpha*(x - loc) / (np.sqrt(2)*scale))))
    return np.log(2) - 0.5*np.log(2*np.pi) + norm_term + skew_term


def get_results(name, JSON_FILE):
    """Get a dataframe 'name' from the results file."""
    # Check if the file exists
    if not os.path.exists(JSON_FILE):
        return None

    with open(JSON_FILE, 'r') as file:
        data = json.load(file)

    json_df = data.get(name)
    if json_df is None:
        return None

    # Convert the JSON object back to DataFrame
    return pd.read_json(json_df)


def update_results(name, dataframe, JSON_FILE, orient="records"):
    """Write a dataframe to the results file."""
    data = {}
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as file:
            data = json.load(file)

    # Convert the DataFrame to JSON and update the dictionary
    data[name] = dataframe.to_json(orient=orient)

    # Write the updated dictionary to the JSON file
    with open(JSON_FILE, 'w') as file:
        json.dump(data, file)


def running_average(data, window_size, kind="mean"):
    """Return running average of size 'window_size'."""
    half_window = window_size // 2
    if kind == "median":
        return np.array([np.median(data[max(0, i - half_window):
            min(len(data), i + half_window + 1)]) for i in range(len(data))])

    return np.array([np.mean(data[max(0, i - half_window):
            min(len(data), i + half_window + 1)]) for i in range(len(data))])


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


def poly(x, *p0):
    """Return polynomial of degree (len(p0) - 1) applied on x."""
    k = len(p0)
    output = np.ones_like(x) * p0[-1]
    for i in range(k - 1):
        output += p0[i] * x**(k - 1 - i)
    return output


def gaus_get_bic(x, y, order, **kwargs):
    """Calculate Bayesian Information Criteria assuming a Gaussian likelihood."""
    if "catch_warnings" in kwargs and kwargs["catch_warnings"]:
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                popt = np.polyfit(x, y, order)
            except np.RankWarning:
                return -np.inf
    else:
        popt = np.polyfit(x, y, order)

    # Get Gaussian MLE and calculate BIC
    residuals = y - poly(x, *popt)
    mu = np.mean(residuals)
    var = np.sum((residuals - mu)**2) / len(residuals)

    n = len(residuals)
    gaussian_lkl = -1. / (2. * var) * np.sum((residuals - mu)**2) -\
        0.5*n*(np.log(var) + np.log(2*np.pi))
    return gaussian_lkl - 0.5*(order+1)*np.log(n), popt, [mu, var]


def bayesian_regularized_linreg(x, y, get_bic=None, f_fit=None,
                                k_min=4, k_pruning=1, max_plateau=2,
                                plot_mean=False, kernel_size=10, disable_tqdm=False,
                                verbose=False, **bic_kwargs):
    """Fit polynomials for different degrees and select the one with the highest BIC."""
    # Organise parameters
    # Default behaviour: fit polynomials and use gaussian likelihood
    get_bic = gaus_get_bic if get_bic is None else get_bic
    f_fit = poly if f_fit is None else f_fit

    # Loop over k
    orders, bics = [], []
    best_bic, best_f_popt, best_distr_popt = -np.inf, None, None

    # Loop until bic stops improving
    i, no_improvement_count = 0, 0
    order = k_min
    pbar = tqdm(disable=disable_tqdm, desc="BIC..", position=1, leave=False)
    while True:
        # Perform the fit
        bic, f_popt, distr_popt = get_bic(x, y, order, **bic_kwargs)
        bics.append(bic)
        orders.append(order)

        if bic > best_bic:
            best_bic = bic
            best_f_popt, best_distr_popt = f_popt, distr_popt
            no_improvement_count = 0  # Reset the counter if there's an improvement
        else:
            no_improvement_count += 1  # Increment the counter if no improvement

        # Check for early stopping
        if no_improvement_count > max_plateau:
            break

        # Loop conditions
        pbar.update(1)
        i += 1
        order += k_pruning
    pbar.close()

    best_fit = f_fit(x, *best_f_popt)
    if verbose:
        # Plot results
        plt.plot(x, y, ".", label="Data", zorder=1)
        plt.plot(x, best_fit, linewidth=2.,
                 label=f"Best fit (k: {orders[np.argmax(bics)]})", zorder=3)
        if plot_mean:
            plt.plot(x, kde_smoothing(y, kernel_size), label="kde", zorder=2)
        plt.legend(loc="best")
        plt.show()

    return best_fit, best_f_popt, best_distr_popt


if __name__ == '__main__':
    # Minimal example
    N = 1000
    x_data = np.linspace(0, np.pi, N)
    model = np.sin(2*np.pi*x_data) + np.sin(np.pi*x_data) + np.random.normal(size=N)
    bayesian_regularized_linreg(x_data, model, verbose=True)
