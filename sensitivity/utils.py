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


def safe_loadtxt(filename, delimiter="\t", usecols=None, **kwargs):
    """Read in a txt file with an arbitrary number of header lines starting with '#'."""
    # Make sure there are no duplicates
    if "skiprows" in kwargs:
        kwargs.pop("skiprows")
    if usecols is None:
        usecols = [0, 1]

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
    return np.loadtxt(filename, delimiter=delimiter, usecols=usecols    ,
                      skiprows=num_header_lines, **kwargs)


def getBinVars(y, nbins, log=False):
    """Return bin delimiters with bin_center and width information."""
    if log:
        bins = np.logspace(np.log10(min(y)), np.log10(max(y)), nbins, base=10)
    else:
        bins = np.linspace(min(y), max(y), nbins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.
    bin_width = bins[1:] - bins[:-1]  # not always a constant!
    return bins, bin_centers, bin_width
