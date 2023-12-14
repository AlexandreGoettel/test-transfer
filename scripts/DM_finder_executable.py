"""Run q0 calculation - condor version."""
import os
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
    parser.add_argument("--rundir", type=str, required=True, help="Arg. file location")
    parser.add_argument("--outdir", type=str, required=True, help="Where to store results.")

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
    assert Y.shape[0] == bkg.shape[0] and Y.shape[1] == peak_shape.shape[0]
    mu_DM = params
    try:
        residuals = Y - np.log(np.exp(bkg) + peak_norm*peak_shape*mu_DM)
    except Exception as err:
        raise err
    log_lkl = sensutils.logpdf_skewnorm(
        residuals, model_args[..., 0], model_args[..., 1], model_args[..., 2])
    assert log_lkl.shape == Y.shape

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
        _Y = Y[i:i+len_peak, :].T
        if _Y.shape[1] < len_peak or 0 in _Y:
            continue

        bkg, peak_norm = [], []
        for j in range(Y.shape[1]):  # For each segment
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
        model_args = reco_model_args[j][i:i+len_peak, :]
        peak_shape.update_freq(frequencies[i])

        bkg, peak_norm, model_args = list(map(np.array, [bkg, peak_norm, model_args]))
        yield _Y, bkg, model_args, peak_norm, peak_shape, frequencies[i]


def get_num_segments(data):
    """Get max N_segments from data keys (created by DM_finder_organiser)."""
    N = -1
    for key in data.keys():
        try:
            num = int(key.split("_")[-1])
        except ValueError:
            continue
        if num > N:
            N = num
    assert N > -1
    return N + 1


def main(rundir="", outdir="", iteration=0, prefix="", n_processes=1, peak_shape_path=None):
    """Find DM using args found in input files path at iteration."""
    # Get args from transferred files
    peak_shape = PeakShape(0, peak_shape_path)
    data = np.load(os.path.join(rundir, f"{prefix}_{iteration}.npz"))

    # Read variables
    Y = data["Y"]
    ifos = data["ifos"]  # 0: L1, 1: H1
    beta_H1, beta_L1 = data["beta_H1"], data["beta_L1"]

    # Decompress background / model args
    N_segments = get_num_segments(data)
    reco_model_args, reco_knots = [], []
    for n_segment in range(N_segments):
        compressed_args, compressed_knots, idx_compressed =\
            list(map(lambda x: data[f"compressed_{x}_{n_segment}"],
                     ["args", "knots", "idx_freq_to_df"]))

        _reco_model_args, _reco_knots = [], []
        for [_, count], args, knots in zip(idx_compressed, compressed_args, compressed_knots):
            _reco_model_args.extend([args]*count)
            last_zero = knots.shape[1] - 1 - np.where(knots[0, :][::-1] == 0)[0][-1]
            _reco_knots.extend([knots[:, :last_zero]]*count)


        reco_model_args.append(np.array(_reco_model_args))
        reco_knots.append(_reco_knots)

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
    np.save(os.path.join(outdir, f"{prefix}_{iteration}_result.npy"), q0_data)


if __name__ == '__main__':
    main(**parse_cmdl_args())

