"""Define models used by the different background-fitting procedures."""
import numpy as np
from scipy.interpolate import interp1d, CubicSpline


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


def model_spline(cube, x_knots, shift=1, extrapolate=False, kind='linear'):
    """Cubic spline fit (to be used in log-log space)."""
    y_knots = cube[shift:len(x_knots)+shift]
    if kind == 'cubic':
        return CubicSpline(x_knots, y_knots, extrapolate=extrapolate)
    else:
        return interp1d(x_knots, y_knots,
                        fill_value=np.nan, bounds_error=not extrapolate)

def model_xy_spline(cube, extrapolate=False):
    """Wrap model_spline with first half of cube is x_knots and rest is y_knots"""
    assert not len(cube) % 2
    n_knots = len(cube) // 2
    if not isinstance(cube, np.ndarray):
        cube = np.array(cube)
    return model_spline(cube[n_knots:], cube[:n_knots],
                        shift=0, kind="cubic", extrapolate=extrapolate)
