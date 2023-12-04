"""Run q0 calculation - condor version."""
import argparse
from multiprocessing import Pool
from tqdm import tqdm, trange
from scipy.optimize import minimize
import numpy as np
# Project imports
import sensutils
import models


def parse_cmdl_args():
    """Parse cmdl args."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--peak-shape-path", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--n-processes", type=int, default=1)

    return vars(parser.parse_args())


class PeakShape(np.ndarray):
    """Hold peak shape (varies over freq.)"""

    def __new__(cls, f, path, dtype=float, buffer=None, offset=0, strides=None, order=None):
        peak_arrays = np.load(path)
        shape = (peak_arrays["data"].shape[1],)
        frequency_bounds = peak_arrays["bounds"]

        obj = super(PeakShape, cls).__new__(cls, shape, dtype, buffer, offset, strides, order)
        obj.f = np.array([f])
        obj.frequency_bounds = frequency_bounds
        obj.peak_shapes = peak_arrays["data"]
        obj._update_array()
        return obj

    def _update_array(self):
        """Update the array based on the value of f."""
        condition = np.where(self.f > self.frequency_bounds)[0]
        idx = 0 if condition.size == 0 else condition[-1] + 1
        np.copyto(self, self.peak_shapes[idx, :])

    def update_freq(self, f):
        """Update the frequency value (and as such the peak shape array if necessary)."""
        self.f = np.array([f])
        self._update_array()


def log_likelihood(params, Y, bkg, peak_norm, peak_shape, model_args):
    """Likelihood of finding dark matter in the data."""
    # Actual likelihood calculation
    mu_DM = params
    try:
        residuals = Y - np.log(np.exp(bkg) + peak_norm*peak_shape*mu_DM)
    except Exception as err:
        raise err
    log_lkl = sensutils.logpdf_skewnorm(
        residuals, model_args[..., 0], model_args[..., 1], model_args[..., 2])

    # Add infinity protection - breaks norm but ok if empirical tests
    mask = np.isinf(log_lkl)
    if np.sum(mask):
        row_mins = np.min(np.where(mask, np.inf, log_lkl), axis=1)
        log_lkl[mask] = np.take(row_mins, np.where(mask)[0])

    return np.sum(np.sum(log_lkl, axis=1))


def get_q0(Y, bkg, model_args, peak_norm, peak_shape,
           min_log10mu=-40, max_log10mu=-32):
    """Do actual q0 related calculations & minimisation, return zero_lkl, max_Lkl, mu_hat."""
    def log_lkl_shape(params):
        return log_likelihood(params[0], Y, bkg, peak_norm, peak_shape, model_args)

    # Get initial guess for mu, based on peak_shape
    test_mus = np.logspace(min_log10mu, max_log10mu, 1000)
    test_lkl = np.array([-log_lkl_shape([mu]) for mu in test_mus])
    mask = np.isnan(test_lkl) | np.isinf(test_lkl)
    if not any(~mask):
        return 0, 0, 0
    initial_guess = test_mus[np.argmin(test_lkl[~mask])]

    # Calculate max lkl
    popt_peak_shape = minimize(
        lambda x: -log_lkl_shape(x),
        [initial_guess],
        bounds=[(0, None)],
        method="Nelder-Mead",
        tol=1e-10)

    # Get max & zero lkl
    zero_lkl = log_lkl_shape([0])
    max_lkl = -popt_peak_shape.fun
    mu = popt_peak_shape.x[0]

    return zero_lkl, max_lkl, mu


def process_q0(args):
    """Wrap calculate_q0 for multiprocessing."""
    Y, bkg, model_args, peak_norm, peak_shape, fmin = args
    try:
        zero_lkl, max_lkl, mu = get_q0(Y, bkg, model_args, peak_norm, peak_shape)
    except AssertionError:
        zero_lkl, max_lkl, mu = np.nan, np.nan, np.nan
    return fmin, zero_lkl, max_lkl, mu


def make_args(Y, frequencies, peak_shape, reco_model_args, reco_knots,
              ifos, beta_H1, beta_L1):
    """Generator for process_q0."""
    len_peak = len(peak_shape)
    for i in trange(len(frequencies) - len_peak, desc="Prep. args"):
        _Y = Y[i:i+len_peak]
        if len(_Y) < len_peak or 0 in _Y:
            continue

        bkg, peak_norm = [], []
        for j in range(_Y.shape[1]):  # For each segment
            # Calculate peak norm
            beta = beta_H1[i:i+len_peak] if ifos[j] == 1 else beta_L1[i:i+len_peak]

            # Get spline from knots
            _bkg = np.zeros(len_peak)
            for k in range(len_peak):
                x_knots, y_knots = reco_knots[j][i+k][0], reco_knots[j][i+k][1]
                _bkg[k] = models.model_xy_spline(
                    np.concatenate([x_knots, y_knots]), extrapolate=True)(
                        np.log(frequencies[i+k])
                    )

            # Fill arrays
            bkg.append(_bkg)
            peak_norm.append(beta)
        model_args = reco_model_args[j, i:i+len_peak, :]
        peak_shape.update_freq(frequencies[i])
        yield _Y, bkg, model_args, peak_norm, peak_shape, frequencies[i]


def main(iteration=0, prefix="", n_processes=1, peak_shape_path=None):
    """Find DM using args found in input files path at iteration."""
    # Get args from transferred files
    peak_shape = PeakShape(0, peak_shape_path)
    data = np.load(f"{prefix}_{iteration}.npz")

    # Read variables
    Y = data["Y"]
    ifos = data["ifos"]  # 0: L1, 1: H1
    beta_H1, beta_L1 = data["beta_H1"], data["beta_L1"]

    # Decompress background / model args
    # compressed_args, compressed_knots, idx_compressed =\
    #     data["model_args"], data["knots"], data["idx_compression"]

    # reco_model_args, reco_knots = [], []
    # for row_idx, row in enumerate(idx_compressed):
    #     expanded_row, expanded_args, expanded_knots = [], [], []

    #     for [val, count], args, knots in zip(row, compressed_args[row_idx],
    #                                          compressed_knots[row_idx]):
    #         expanded_row.extend(np.full(count, val))
    #         expanded_args.extend([args] * count)
    #         last_zero = knots.shape[1] - 1 - np.where(knots[0, :][::-1] == 0)[0][-1]
    #         expanded_knots.extend([knots[:, :last_zero]]*count)

    #     reco_model_args.append(expanded_args)
    #     reco_knots.append(expanded_knots)
    # reco_model_args = np.array(reco_model_args)
    reco_knots, reco_model_args = data["knots"], data["model_args"]

    # Derived variables
    len_peak = len(peak_shape)
    frequencies = np.logspace(np.log10(data["fmin"]), np.log10(data["fmax"]), Y.shape[0])

    # Now start parallel q0 calculation
    results = []
    _args = list(make_args(Y, frequencies, peak_shape, reco_model_args,
                           reco_knots, ifos, beta_H1, beta_L1))
    print("Starting q0 calculation..")
    with Pool(n_processes) as pool,\
            tqdm(total=len(frequencies) - len_peak, position=0,
                 desc="q0 calc.", leave=True) as pbar:
        for result in pool.imap(process_q0, (arg for arg in _args), chunksize=10):
            results.append(result)
            pbar.update(1)

    # Merge results
    q0_data = np.zeros((len(results), 4))
    for i, result in enumerate(results):
        q0_data[i, :] = result  # freqs, zero_lkl, max_lkl, mu
    np.save(f"{prefix}_{iteration}_result.npy", q0_data)

    # Debugging
    from matplotlib import pyplot as plt
    pos_mu = q0_data[:, 3] > 0
    q0 = np.zeros(q0_data.shape[0])
    q0[pos_mu] = -2*(q0_data[:,   1][pos_mu] - q0_data[:, 2][pos_mu])

    # q0 vs frequency
    ax = plt.subplot(111)
    ax.set_xscale("log")
    idx = np.argsort(q0_data[:, 0])
    ax.plot(q0_data[idx, 0], np.sqrt(q0[idx]))
    ax.set_title("Z")
    ax.set_ylim(0, ax.get_ylim()[1])

    # hat(mu) vs frequency
    plt.figure()
    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(q0_data[idx, 0], np.sqrt(q0_data[idx, 3]), zorder=1)
    ax.axhline(1e-17, color="r", linestyle="--", zorder=2, linewidth=2)
    ax.set_title("Lambda_i^-1")

    # Clean data
    freq = q0_data[idx, 0]
    diff = freq[1:] / freq[:-1]
    plt.figure()
    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    bins = np.logspace(np.log10(min(diff)), np.log10(max(diff)), 100)
    ax.hist(diff, bins)
    _lim = 1.0002
    ax.axvline(_lim, color="r", linestyle="--")

    # Take only highest Z value per block
    sorted_freq, mus = q0_data[idx, 0], np.sqrt(q0_data[idx, 3])
    previous_f = sorted_freq[0]
    group_f, group_Z, group_mu = [previous_f], [np.sqrt(q0[idx][0])], [mus[0]]
    plot_f, plot_Z, plot_mu = [], [], []
    for i, val in enumerate(q0[idx]):
        if i == 0:
            continue
        f_i = sorted_freq[i]

        if f_i / previous_f > _lim:
            # Start new group
            idx_peak = np.argmax(group_Z)
            plot_Z.append(group_Z[idx_peak])
            plot_f.append(group_f[idx_peak])
            plot_mu.append(group_mu[idx_peak])
            group_f, group_Z, group_mu = [f_i], [np.sqrt(val)], [mus[i]]
        else:
            # Continue to fill group
            group_f.append(f_i)
            group_Z.append(np.sqrt(val))
            group_mu.append(mus[i])

        # Loop condition
        previous_f = f_i

    [plot_f, plot_Z, plot_mu] = list(map(np.array, [plot_f, plot_Z, plot_mu]))
    _lim_Z = 25
    plt.figure()
    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.scatter(plot_f, plot_Z)
    ax.axhline(_lim_Z, color="r", linestyle="--")
    ax.set_title("Cleaned Z")

    plt.figure()
    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    mask = plot_Z > _lim_Z
    ax.scatter(plot_f[mask], plot_mu[mask])
    ax.set_title("Cleaned Lambda")
    ax.grid(color="grey", alpha=.4, linestyle="--", which="both")
    plt.show()

if __name__ == '__main__':
    main(**parse_cmdl_args())
