"""Collection of useful LPSD resources."""
import numpy as np


class LPSDVars:
    """Hold base LPSD vars & derive process variables."""

    def __init__(self, fmin, fmax, resolution, fs, epsilon):
        self.fmin, self.fmax, self.resolution, self.fs, self.epsilon =\
            fmin, fmax, resolution, fs, epsilon
        # Derived variables
        self.g = np.log(fmax) - np.log(fmin)
        self.Jdes = int(2 + self.g / np.log(1. + self.resolution))

    def N(self, j):
        """Get the segment length at bin position j."""
        return self.fs/self.fmin * np.exp(-j*self.g / (self.Jdes - 1.)) /\
            (np.exp(self.g / (self.Jdes - 1.)) - 1.)

    def j(self, j0):
        """Block approximation: get the next block position starting at j0."""
        Nj0 = self.N(j0)
        return - (self.Jdes - 1.) / self.g * np.log(Nj0*(1 - self.epsilon/100.) *\
            self.fmin/self.fs * (np.exp(self.g / (self.Jdes - 1.)) - 1.))
