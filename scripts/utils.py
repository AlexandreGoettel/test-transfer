"""Collection of useful LPSD resources."""
import numpy as np


class LPSDVars:
    """Hold base LPSD vars & derive process variables."""

    def __init__(self, fmin, fmax, resolution):
        self.fmin, self.fmax, self.resolution = fmin, fmax, resolution
        self.g = np.log(fmax) - np.log(fmin)

    @property
    def Jdes(self):
        """Get the number of required frequency bins as interger, rounded up."""
        return int(2 + self.g / np.log(1. + self.resolution))
