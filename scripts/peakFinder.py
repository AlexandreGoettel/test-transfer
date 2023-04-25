"""Class to find peaks in LPSD output."""
import csv
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit as fit
from tqdm import tqdm
import utils
import hist


class PeakFinder:
    """Find peaks in LPSD output."""
    def __init__(self, **kwargs):
        """Set required attributes and store output file contents as ndarray."""
        required_attrs = ["fs", "fmin", "fmax", "resolution", "epsilon", "name"]
        for name, value in kwargs.items():
            if name in required_attrs:
                required_attrs.remove(name)
                setattr(self, name, value)
            else:
                print(f"[PeakFinder] Unknown parameter '{name}', ignoring..")
        # Check if all required attributes were set
        if len(required_attrs) > 0:
            raise ValueError("Missing attributes for PeakFinder.init: " + ", ".join(required_attrs))     
        
        # Calculate useful variables
        self.J = utils.Jdes(self.fmin, self.fmax, self.resolution)
        self.g = np.log(self.fmax) - np.log(self.fmin)
        
        # Read data
        self.freq, self.psd = self.read(self.name)

    def read(self, name, dtype=np.float64):
        """
        Read an output file from LPSD. Needs self.J to exist.
        
        name(str): output file name.
        return: frequency & PSD arrays.
        """
        x, y = [], []
        with open(name, "r") as _file:
            data = csv.reader(_file, delimiter="\t")
            print("Reading input PSD..")
            for i, row in enumerate(tqdm(data, total=self.J)):
                try:
                    x += [float(row[0])]
                    y += [float(row[1])]
                except ValueError:
                    continue
        return np.array(x, dtype=dtype), np.array(y, dtype=dtype)
    
    def block_position_gen(self):
        """Generator for block positions."""
        j0 = 0
        j = utils.j(j0, self.J, self.g, self.epsilon, self.fmin, self.fmax, self.fs, self.resolution)
        yield 0  # Always start at 0
        yield int(j)
        
        while True:
            j0 = j
            j = utils.j(j0, self.J, self.g, self.epsilon, self.fmin, self.fmax, self.fs, self.resolution)
            if j >= self.J - 1:
                yield self.J - 1
                return
            yield int(j)

    def baseline_fit(self, pos0, pos1, nbins=100, verbose=False):
        """
        Fit a skew gaussian to log(y) projection of the PSD data in the segment.
        
        Assume negligible correlations and output popt, sqrt(diag(pcov)), and reduced chi_sqr.
        """
        x, y = self.freq[pos0:pos1], self.psd[pos0:pos1]
        
        # Transform to Y = ln(y)
        Y = np.log(y)
        
        # Fit a skew gaussian
        bins, bin_centers, bin_width = utils.getBinVars(Y, nbins, log=False)
        h0, bins = np.histogram(Y, bins)
        _argmax = np.argmax(h0 / bin_width)
        p0 = [10*h0[_argmax], bin_centers[_argmax], np.std(Y, ddof=1), 0]
        popt, pcov = hist.fit_hist(hist.skew_gaus, Y, bins, p0=p0)
        
        # Calculate chi_sqr
        y = hist.skew_gaus(bin_centers, *popt)
        chi_sqr = utils.getChiSquarePoisson(y*bin_width, h0, ddof=4)
    
        if verbose:
            ax = hist.plot_func_hist(hist.skew_gaus, popt, Y, bins)
            plt.show()
        return popt, np.sqrt(np.diag(pcov)), chi_sqr
    
    def baseline_fit_cutoff(self, pos0, pos1, nbins=100, cutoff=0.1):
        """
        Fit a skew gaussian to log(y-proj) but include a cut-off for protection.
        
        Assume negligible correlations and output popt, sqrt(diag(pcov)), and reduced chi_sqr.
        """
        x, y = self.freq[pos0:pos1], self.psd[pos0:pos1]
        
        # Transform to Y = ln(y)
        Y = np.log(y)
        
        # Prepare hist fit
        bins, bin_centers, bin_width = utils.getBinVars(Y, nbins, log=False)
        h0, bins = np.histogram(Y, bins)
        _argmax = np.argmax(h0 / bin_width)
        p0 = [10*h0[_argmax], bin_centers[_argmax], np.std(Y, ddof=1), 0]
        
        # Apply cut-off
        m = h0 > 0.1*h0[_argmax]
        h0, bin_centers, bin_width = h0[m], bin_centers[m], bin_width[m]
        assert len(h0) > 4
        
        # Perform a custom proper chi square fit  # TODO: incorporate in hist.py?
        f = hist.skew_gaus
        m = h0 == 0
        if np.sum(np.array(m, dtype=int)):
            zero_h0, zero_bin_centers, zero_bin_width = h0[~m], bin_centers[~m], bin_width[~m]
            p0, _ = fit(f, zero_bin_centers, zero_h0/zero_bin_width, p0=p0,
                        sigma=np.sqrt(zero_h0)/zero_bin_width, absolute_sigma=True)
            h0 = np.array(h0, dtype=np.float64)
            h0[m] = f(bin_centers[m], *p0) * bin_width[m]
        
        # Second fit using output of previous one
        popt, pcov = fit(f, bin_centers, h0/bin_width, p0=p0,
                            sigma=np.sqrt(h0)/bin_width, absolute_sigma=True)
        
        # Calculate chi_sqr
        y = f(bin_centers, *popt)
        chi_sqr = utils.getChiSquarePoisson(y*bin_width, h0, ddof=4)   
        return popt, np.sqrt(np.diag(pcov)), chi_sqr
