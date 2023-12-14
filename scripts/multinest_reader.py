"""Implement class to analyse MultiNest results and make life easier."""
import json
import subprocess
import shlex
import numpy as np
from matplotlib import pyplot as plt
from pymultinest.analyse import Analyzer


class MultiNestAnalyser:
    """Look at and analyse results from a MultiNest run."""

    def __init__(self, out):
        """Set the out folder and get the parameter info."""
        self.out = out
        self.parameters = json.load(open(out + "params.json"))
        self.nParams = len(self.parameters)
        # Get useful data for later
        a = Analyzer(self.nParams, outputfiles_basename=self.out)
        self.data = a.get_data()
        self.stats = a.get_stats()

    def makeMarginalPlots(self):
        """Run multinest_marginals_fancy on the MultiNest output."""
        cmd = "multinest_marginals_fancy.py {}".format(self.out)
        print("Preparing marginal plots, please wait..")
        p = subprocess.Popen(shlex.split(cmd))
        p.wait()
        print("All done!")

    def getResults(self):
        """Return mean, std, best-fit of the marginalized posteriors."""
        mode = self.stats["modes"][0]

        mean = mode["mean"]
        sigma = mode["sigma"]
        bestfit = mode["maximum a posterior"]

        return mean, sigma, bestfit

    def getSamplesWeights(self):
        """Get samples and format them into physical variables with weights."""
        # Format data into physical values
        i = self.data[:, 1].argsort()[::-1]
        samples, weights, loglike = self.data[i, 2:], self.data[i, 0],\
            self.data[i, 1]
        Z = self.stats["global evidence"]
        logvol = np.log(weights) + 0.5*loglike + Z
        logvol -= np.max(logvol)

        return samples, weights, logvol, loglike

    def plotPosteriors(self):
        """Make histograms of the posterior distributions."""
        samples, weights, _, loglike = self.getSamplesWeights()

        # One plot for each parameter
        for i in range(0, self.nParams):
            x = samples[:, i]
            plt.figure()
            plt.hist(x, 50, weights=weights)
            plt.title(self.parameters[i])
        plt.show()

    def getQuantiles(self, quantiles):
        """Return quantiles of the posterior distributions."""
        samples, weights, _, loglike = self.getSamplesWeights()

        output = np.zeros((len(quantiles), self.nParams))
        for i in range(0, self.nParams):
            x = samples[:, i]
            idx = np.argsort(x)

            # Sort weights along samples
            sortedWeights = weights[idx]
            # Compute normalised cdf
            cdf = np.cumsum(sortedWeights)[:-1]
            cdf /= cdf[-1]
            cdf = np.append(0, cdf)
            # Get quantiles along the weights, apply to x
            output[:, i] = np.interp(quantiles, cdf, x[idx])

        return output

    def getNestedEvidence(self):
        """Return the nested sampling global evidence."""
        Z = self.stats["nested sampling global log-evidence"]
        return Z

    def getGlobalEvidence(self):
        """Return the (importance) global evidence with sigma."""
        return self.stats["global evidence"],\
            self.stats["global evidence error"]
