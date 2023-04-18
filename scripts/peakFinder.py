"""Class to find peaks in LPSD output."""
import csv
import numpy as np
import utils


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
        
        # Read data
        self.freq, self.psd = self.read(self.name)
        
        # Calculate useful variables
        self.J = utils.Jdes(self.fmin, self.fmax, self.resolution)
        self.g = np.log(self.fmax) - np.log(self.fmin)

    def read(self, name, dtype=np.float64):
        """
        Read an output file from LPSD.
        
        name(str): output file name.
        
        return: frequency & PSD arrays.
        """
        x, y = [], []
        with open(name, "r") as _file:
            data = csv.reader(_file, delimiter="\t")
            for i, row in enumerate(data):
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
        yield int(j)
        
        while True:
            j0 = j
            j = utils.j(j0, self.J, self.g, self.epsilon, self.fmin, self.fmax, self.fs, self.resolution)
            if j >= self.J - 1:
                yield self.J - 1
                return
            yield int(j)
