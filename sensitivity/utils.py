"""Collect useful functions in LPSD calculations."""
import numpy as np


def get_Jdes(fmin, fmax, resolution):
    """Return the required number of bins."""
    g = np.log(fmax) - np.log(fmin)
    return int(1 + g/(np.log(fmin + fmin*resolution) - np.log(fmin)))


def get_fmin(length, resolution):
    """Return fmin in Hz given length in s."""
    return 1. / (length * resolution)
