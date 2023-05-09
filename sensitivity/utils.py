"""Collect useful functions in LPSD calculations."""
import os
import numpy as np


def get_Jdes(fmin, fmax, resolution):
    """Return the required number of bins."""
    g = np.log(fmax) - np.log(fmin)
    return int(1 + g/(np.log(fmin + fmin*resolution) - np.log(fmin)))


def get_fmin(length, resolution):
    """Return fmin in Hz given length in s."""
    return 1. / (length * resolution)


def safe_loadtxt(filename, delimiter="\t", **kwargs):
    """Read in a txt file with an arbitrary number of header lines starting with '#'."""
    # Make sure there are no duplicates
    if "skiprows" in kwargs:
        kwargs.pop("skiprows")

    # Abs. vs rel. path
    if not filename.startswith("/"):
        filename = os.path.join(os.getcwd(), filename)

    # Count the number of header lines
    num_header_lines = 0
    with open(filename, 'r') as f:
        for line in f.readlines():
            if not line.startswith('#'):
                break
            num_header_lines += 1

    # Load the data using np.loadtxt(), skipping the header lines
    return np.loadtxt(filename, delimiter=delimiter,
                      skiprows=num_header_lines, **kwargs)
