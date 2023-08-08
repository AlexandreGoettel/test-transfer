"""Implement model functions for multinest/ultranest models in background fits."""
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import norm, skewnorm
import sensutils


# Base models
def model_segment_line(cube, x_data, block_positions, shift=0):
    """Model line on all segments."""
    output = np.zeros_like(x_data)
    for i, start_pos in enumerate(block_positions[:-1]):
        end_pos = block_positions[i + 1]
        a, b = cube[2*i+shift:2*(i+1)+shift]
        output[start_pos:end_pos] += a * x_data[start_pos:end_pos] + b
    return output


def model_segment_slope(cube, x_data, block_positions, shift=0):
    """Model slope-only line on all segments based on spline bkg."""
    y_model = np.zeros_like(x_data)
    for i, start_pos in enumerate(block_positions[:-1]):
        end_pos = block_positions[i + 1]
        # Define line
        a = cube[i+shift]
        offset = -a*x_data[start_pos]
        y_model[start_pos:end_pos] += a*x_data[start_pos:end_pos] + offset

    return y_model


def model_segment_line_vec(cube, x_data, block_positions, shift=0):
    """Model line on all segments."""
    # Add a new dimension if cube is 1D
    cube = cube[np.newaxis, :] if cube.ndim == 1 else cube
    output = np.zeros((cube.shape[0], x_data.shape[0]))
    for i, (start_pos, end_pos) in enumerate(zip(block_positions[:-1], block_positions[1:])):
        a, b = cube[:, 2*i+shift:2*(i+1)+shift].T
        output[:, start_pos:end_pos] += (a[:, np.newaxis] * x_data[start_pos:end_pos]) + b[:, np.newaxis]
    return output


# Interference model
def model_interference(cube, x_data, y_data, block_positions,
                       x_knots, segment_model):
    """Fit segments for given spline params."""
    y_spline = model_spline(cube, x_knots, shift=0)(x_data)
    popt, _ = segment_model(x_data, y_data, x_knots, cube,
                            block_positions, verbose=False)

    return y_spline + model_segment_line(
        popt.flatten(), x_data, block_positions, shift=0)


def likelihood_interference(cube, x, y_data, sigma, x_knots,
                            block_positions, segment_model):
    """Implement gaussian likelihood for combined fit."""
    y_model = model_interference(cube, x, y_data, block_positions,
                                 x_knots, segment_model)
    sigma = np.abs(sigma*y_data[0])
    lkl = -np.sum(0.9189385 + np.log(sigma) +
                  0.5*((y_model - y_data) / sigma)**2)
    return lkl


def prior_interference(cube, y_knots, y_sigma):
    """Implement spline priors with no shift."""
    params = cube.copy()
    n_sigma = 3
    for i, (k, sigma) in enumerate(zip(y_knots, y_sigma)):
        # Uniform prior:
        lo, hi = k - n_sigma*sigma, k + n_sigma*sigma
        params[i] = lo + cube[i] * (hi - lo)
    return params


# Simple spline model
def model_spline(cube, x_knots, shift=1, extrapolate=False):
    """Cubic spline fit to be used in log-log space."""
    y_knots = cube[shift:len(x_knots)+shift]
    return CubicSpline(x_knots, y_knots, extrapolate=extrapolate)
    # return interp1d(x_knots, y_knots,
    #                 fill_value=np.nan, bounds_error=not extrapolate)


def model_spline_vec(cube, x_knots, shift=1):
    """Cubic spline fit to be used in log-log space."""
    N, _ = cube.shape
    y_knots = cube[:, shift:shift + len(x_knots)]
    splines = [CubicSpline(x_knots, y_knots[i],
                           extrapolate=False) for i in range(N)]
    return splines


def likelihood_simple_spline(cube, x, y_data, x_knots):
    """Implement log-gaussian likelihood."""
    y_model = model_spline(cube, x_knots)(x)
    sigma = cube[0]
    return -np.sum(0.9189385 + np.log(sigma) + 0.5*((y_model - y_data) / (sigma*y_data[0]))**2)


def prior_simple_spline(cube, x, y, x_knots, premade=True):
    """Implement priors for simplified spline-only fit."""
    # TODO: Add option to recover numbers from x, y in case of data change
    # Careful: that's extremely slow, only do it once then pass fast prior
    # Relative prior on sigma
    lo_log, hi_log = -4, 0
    cube[0] = 10**(lo_log + cube[0] * (hi_log - lo_log))

    # Knots
    if premade:
        lows = [-93.60771937700463, -100.15027347174316, -101.4989203610385, -102.16893223292914,
                -105.60664708979223, -108.22471150838881, -129.65920003576355]
        highs = [-88.3506897916026, -95.9377992860845, -99.54901756625651, -101.98191640896727,
                -105.33045493809118, -106.38279029533324, -128.50266685451925]
    else:
        lows = [min(y) for x in x_knots]
        highs = [max(y) for x in x_knots]
    for i in range(len(x_knots)):
        # Set from y data around this position
        lo, hi = lows[i], highs[i]
        cube[i+1] = lo + cube[i+1] * (hi - lo)


# Full combined model
def model_combined(cube, x_data, block_positions, x_knots):
    """Wrap for full bkg PSD model."""
    return model_spline(cube, x_knots, shift=0)(x_data) +\
        model_segment_line(cube, x_data, block_positions, shift=len(x_knots))


def model_combined_slope(cube, x_data, y_data, block_positions, x_knots, buffer=20):
    """Full bkg PSD model with "smart" slope reduction."""
    y_spline = model_spline(cube, x_knots, shift=0)(x_data)
    y_lines = model_segment_slope(cube, x_data, y_spline, block_positions,
                                  shift=len(x_knots), buffer=buffer)
    return y_spline + y_lines


def model_combined_vec(cube, x_data, block_positions, x_knots):
    """Wrap for full bkg PSD model."""
    splines = model_spline_vec(cube, x_knots, shift=0)
    output = np.zeros_like(x_data)
    for i in range(cube.shape[0]):
        output += splines[i](x_data)
    return output + model_segment_line_vec(cube, x_data, block_positions, shift=len(x_knots))


def likelihood_combined(cube, x, y_data, x_knots, block_positions):
    """Implement gaussian likelihood for combined fit."""
    y_model = model_combined(cube, x, block_positions, x_knots)
    sigma = np.abs(7e-2*y_data[0])  # TODO hardcoded?
    lkl = -np.sum(0.9189385 + np.log(sigma) +
                  0.5*((y_model - y_data) / sigma)**2)
    return lkl


def likelihood_combined_skew_slope(cube, x, y_data, x_knots, block_positions, slope_buffer=20):
    """Skew norm likelihood for slope-only combined fit."""
    y_model = model_combined_slope(cube, x, y_data, block_positions, x_knots)
    lkl = (0.9189385 + 0.5*((y_model - y_data))**2)
    # lkl = -2*np.log(np.abs(y_data - y_model))
    lkl[np.isinf(lkl)] = -10000
    return lkl.sum() / len(lkl)


def likelihood_combined_vec(cube, x, y_data, x_knots, block_positions):
    """Implement gaussian likelihood for combined fit."""
    y_model = model_combined_vec(cube, x, block_positions, x_knots)
    sigma = np.abs(7e-2*y_data[0])  # TODO hardcoded?
    # ensure that y_data, y_model, and sigma are of the same shape before operation
    # y_data = y_data[:, np.newaxis] if y_data.ndim == 1 else y_data
    sigma = sigma * np.ones_like(y_model)  # ensure sigma is broadcasted correctly
    lkl = -np.sum(0.9189385 + np.log(sigma) + 0.5 * ((y_model - y_data) / sigma)**2,
                  axis=1)
    return lkl


def prior_combined(cube, y_knots, y_sigma, popt_segments, pcov_segments,
                   n_sigma=1):
    """
    Implement priors for combined fit based on previous fits.

    cube[:n_knots] is knot height in log space
    cube[n_knots:] is line parameters a,b,a,b,..
    """
    # Priors on spline params
    param = cube.copy()
    for i, (k, sigma) in enumerate(zip(y_knots, y_sigma)):
        # Uniform prior:
        lo, hi = k - n_sigma*sigma, k + n_sigma*sigma
        param[i] = lo + cube[i] * (hi - lo)
        # Gaussian prior:
        # cube[i] = norm.ppf(cube[i], loc=k, scale=sigma)

    # Priors on segment lines
    for i in range(popt_segments.shape[0]):
        for j in range(popt_segments.shape[1]):  # a, b
            mu = popt_segments[i, j]
            sigma = np.sqrt(np.diag(pcov_segments[i, ...])[j])
            # Uniform prior:
            lo, hi = mu - n_sigma*sigma, mu + n_sigma*sigma
            param[2*i+j+len(y_knots)] = lo + cube[2*i+j+len(y_knots)] * (hi - lo)
            # Gaussian prior:
            # idx = 2*i + j + len(y_knots)
            # cube[idx] = norm.ppf(cube[i], loc=mu, scale=sigma)
    return param


def prior_combined_vec(cube, y_knots, y_sigma, popt_segments, pcov_segments):
    """
    Implement priors for combined fit based on previous fits.

    cube[:n_knots] is knot height in log space
    cube[n_knots:] is line parameters a,b,a,b,..
    """
    cube = cube[np.newaxis, :] if cube.ndim == 1 else cube
    param = cube.copy()

    n_sigma = 1

    # Priors on spline params
    lo = y_knots - n_sigma*y_sigma
    hi = y_knots + n_sigma*y_sigma
    param[:, :len(y_knots)] = lo + cube[:, :len(y_knots)] * (hi - lo)

    # Priors on segment lines
    mu = popt_segments.reshape(-1)
    sigma = np.sqrt(pcov_segments.diagonal(0, 1, 2).reshape(-1))

    lo = mu - n_sigma*sigma
    hi = mu + n_sigma*sigma
    param[:, len(y_knots):] = lo + cube[:, len(y_knots):] * (hi - lo)
    return param
