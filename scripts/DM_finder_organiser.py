"""
Companion file to DM_finder_v3.py to generate files to transfer in condor.
"""
import os
import glob
import argparse
from tqdm import tqdm, trange
import h5py
from scipy.interpolate import interp1d
from scipy.stats import skewnorm
from scipy import constants
import pandas as pd
import numpy as np
# Project imports
import utils
import sensutils
import models


BASE_PATH = os.path.split(os.path.abspath(__file__))[0]


def parse_cmdl_args():
    """Parse cmdl args to pass to main."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add arguments
    parser.add_argument("--prefix", type=str, required=True,
                        help="Where to save the data (for condor).")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the MC/data input file.")
    parser.add_argument("--data-prefix", type=str, default="result",
                        help="Prefix of data files.")
    parser.add_argument("--json-path", type=str, required=False,
                        help="Path to the post-processing json file.")
    parser.add_argument("--peak-shape-path", type=str, default=None,
                        help="Path to the peak_shape_data.npz file.")
    parser.add_argument("--injection-path", type=str, default=None,
                        help="If path is given, add those specified injections using peak shape.")
    parser.add_argument("--injection-peak-shape-path", type=str, default=None,
                        help="If given, use this for injections instead of peak-shape-path.")
    parser.add_argument("--dname", type=str, required=True, help="Name of the PSD dataset.")
    parser.add_argument("--dname-freq", type=str, required=True,
                        help="Name of the frequency dataset.")
    parser.add_argument("--max-chi", type=int, default=10,
                        help="Maximum chi^2 deviation to skew norm fit in a chunk.")
    parser.add_argument("--fmin", type=float, default=10)
    parser.add_argument("--fmax", type=float, default=5000)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--isMC", action="store_true")
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


class DMFinder:
    """Hold DM-finding related options."""

    def __init__(self, data_path=None, json_path=None, dname=None,
                 peak_shape_path=None, dname_freq=None,
                 injection_path=None, injection_peak_shape_path=None,
                 max_chi=10, data_prefix="result", **kwargs):
        """Initialise necessarily shared variables and prep data."""
        # Init variables
        self.kwargs = kwargs
        self.max_chi = max_chi
        self.injection_path = injection_path
        self.rho_local = 0.4 / (constants.hbar / constants.e
                                * 1e-7 * constants.c)**3  # GeV/cm^3 to GeV^4

        # Generate peak shape
        self.peak_shape = None if peak_shape_path is None else PeakShape(0, peak_shape_path)
        self.injection_peak_shape = self.peak_shape if injection_peak_shape_path is None else\
            PeakShape(0, injection_peak_shape_path)
        self.len_peak = 0 if self.peak_shape is None else len(self.peak_shape)

        # Read data & TF info
        self.transfer_functions, self.f_A_star = self.get_f_A_star()
        self.data_info = self.get_info_from_data_path(data_path, json_path,
                                                      prefix=data_prefix)
        self.dfile = self.dfile_from_data(data_path, dname, dname_freq)
        self.dset = self.dfile[dname]
        self.freqs = self.dfile[dname_freq]

    def __del__(self):
        """Close HDF properly."""
        if hasattr(self, "data_info"):
            for dfile, _, _ in self.data_info:
                dfile.close()
        if hasattr(self, "dfile") and self.dfile:
            self.dfile.close()

    def parse_ifo(self, i):
        """Parse the ifo from the dset attr at index i."""
        return self.dset.attrs[str(i)].split("_")[-1]

    def get_info_from_data_path(self, data_path, json_path, prefix="result"):
        """Get all possible info from result files in data_path and json_path."""
        data_paths = sorted(list(glob.glob(os.path.join(data_path, f"{prefix}*"))))
        data_info = []
        for _data_path in tqdm(data_paths, desc="Opening data HDFs", leave=False):
            _data_info = []
            _data_info.append(h5py.File(sensutils.get_corrected_path(_data_path), "r"))
            ifo = sensutils.parse_ifo(_data_path)
            _data_info.append(ifo)
            df_key = "splines_" + sensutils.get_df_key(_data_path)
            _data_info.append(sensutils.get_results(df_key, json_path))
            assert _data_info[-1] is not None
            # _data_info.append(df_key)
            data_info.append(_data_info)

        return data_info

    def dfile_from_data(self, data_path, dname, dname_freq):
        """Create dfile by combining available data through data_info."""
        dfile = h5py.File(os.path.join(data_path, "tmp_orga.h5"), "w")
        # Get PSD/freq shape
        # Using min because the difference in the frequencies between the different files
        # should only be a single bin anyway (last bin)
        psd_length = min([len(inf[0][dname_freq]) for inf in self.data_info])
        dset = dfile.create_dataset(dname, (psd_length, len(self.data_info)),
                                    dtype=np.float64)
        # Get frequencies
        for inf in self.data_info:
            if len(inf[0][dname_freq]) == psd_length:
                dfile.create_dataset(dname_freq, data=inf[0][dname_freq][:])
                break
        # Metadata
        bkg = dfile.create_dataset("bkg", (psd_length, len(self.data_info)),
                                   dtype=np.float64)
        model_args = dfile.create_dataset("model_args", (psd_length, len(self.data_info), 3),
                                          dtype=np.float64)

        # Fill data dset
        df_columns = ["fmin", "fmax", "x_knots", "y_knots",
                      "alpha_skew", "loc_skew", "sigma_skew", "chi_sqr"]
        knots_collection = [[[], []]]*len(self.data_info)
        N_segments = len(self.data_info)
        max_N_data = max(len(info[2]) for info in self.data_info)
        max_N_knots = max(max(len(knots) for knots in df["x_knots"])
                          for [_, _, df] in self.data_info)
        knots_collection = np.zeros((N_segments, 2, max_N_data, max_N_knots))
        for i, (hdf_path, ifo, df) in enumerate(
                tqdm(self.data_info, desc="Fill dset.", leave=False)):
            dset[:, i] = hdf_path["logPSD"][:psd_length]
            dset.attrs[str(i)] = ifo

            for j, row in df.iterrows():
                fmin, fmax, x_knots, y_knots, alpha_skew, loc_skew, sigma_skew, chi_sqr\
                    = row[df_columns]
                knots_collection[i, 0, j, :len(x_knots)] = x_knots
                knots_collection[i, 1, j, :len(x_knots)] = y_knots
                if chi_sqr >= self.max_chi:
                    continue

                # Get frequency indices
                frequencies = hdf_path["frequency"]
                idx_min = sensutils.binary_search(frequencies, fmin)
                idx_max = sensutils.binary_search(frequencies, fmax)

                # Get backgound model
                spline = models.model_xy_spline(
                    np.concatenate([x_knots, y_knots]), extrapolate=True)(
                        np.log(frequencies[idx_min:idx_max+1]))

                # Write to disk
                bkg[idx_min:idx_max+1, i] = spline
                for k, val in enumerate([alpha_skew, loc_skew, sigma_skew]):
                    model_args[idx_min:idx_max+1, i, k] = val

        dfile.create_dataset("knots", data=knots_collection, dtype=np.float64)
        # Clean dataset (loop to avoid excessive memory usage)
        for j in trange(dset.shape[1], desc="Cleaning", leave=False):
            dset[dset[:, j] == 0, j] = np.nan

        return dfile

    def get_f_A_star(self):
        """Get dicts of transfer function info."""
        tf_dir = os.path.join(BASE_PATH, "data", "transfer_functions")
        transfer_functions = {}
        transfer_functions["H1"] = pd.read_csv(os.path.join(tf_dir, "Amp_Cal_LHO.txt"),
                                            delimiter="\t")
        transfer_functions["L1"] = pd.read_csv(os.path.join(tf_dir, "Amp_Cal_LLO.txt"),
                                            delimiter="\t")
        f_A_star = {"H1": interp1d(transfer_functions["H1"]["Freq_o"],
                                transfer_functions["H1"]["Amp_Cal_LHO"]),
                    "L1": interp1d(transfer_functions["L1"]["Freq_Cal"],
                                transfer_functions["L1"]["amp_cal_LLO"])}
        return transfer_functions, f_A_star


def make_args(fndr, fmin, fmax, isMC=False):
    """Generator to q0 calculation args between fmin and fmax."""
    frequencies = fndr.freqs
    idx_min = sensutils.binary_search(frequencies, fmin)
    idx_max = 1 + sensutils.binary_search(frequencies, fmax)

    freq_Hz = frequencies[idx_min:idx_max]
    beta_H1 = fndr.rho_local / (np.pi*freq_Hz**3 * fndr.f_A_star["H1"](freq_Hz)**2
                                * (constants.e / constants.h)**2)
    beta_L1 = fndr.rho_local / (np.pi*freq_Hz**3 * fndr.f_A_star["L1"](freq_Hz)**2
                                * (constants.e / constants.h)**2)

    # Prepare background arguments
    N_data = fndr.dset.shape[1]
    max_N_knots = max(max(len(knots) for knots in df["x_knots"])
                          for [_, _, df] in fndr.data_info)
    N_freqs = len(freq_Hz)
    unique_knots = np.zeros((N_data, N_freqs, 2, max_N_knots))
    unique_model_args = np.zeros((N_data, N_freqs, 3))
    idx_freq_to_df = np.zeros((N_data, N_freqs), dtype=np.int32)
    ifos = np.zeros(N_data, dtype=np.int16)

    # Fill arrays
    Y = np.zeros((idx_max-idx_min, fndr.dset.shape[1])) if isMC else\
        fndr.dset[idx_min:idx_max, :]
    for i, (_, _, df) in enumerate(fndr.data_info):
        ifo = fndr.dset.attrs[str(i)]
        ifos[i] = 0 if ifo == "L1" else 1
        # Find the df entries that contains the tested frequency
        mask = (df['fmin'] <= fmax) & (df['fmax'] >= fmin)
        _N_dfs = np.sum(np.array(mask, dtype=int))
        # Skip if frequencies are not represented here
        if not _N_dfs:
            continue

        # For each idx in Y, get j_idx
        for j, row in df.iterrows():
            condition = (row["fmin"] <= freq_Hz) & (row["fmax"] > freq_Hz)
            idx_freq_to_df[i, condition] = j

            # Get knots
            x_knots, y_knots = np.array(row["x_knots"]), np.array(row["y_knots"])
            k = len(x_knots)
            unique_knots[i, condition, 0, :k] = x_knots.reshape((1, k))
            unique_knots[i, condition, 1, :k] = y_knots.reshape((1, k))

            # Get model args
            alpha, loc, sigma = row["alpha_skew"], row["loc_skew"], row["sigma_skew"]
            unique_model_args[i, condition, :] = np.array([alpha, loc, sigma]).reshape((1, 3))

            # Build logPSD data
            if isMC:
                spline = models.model_xy_spline(
                    np.concatenate([x_knots, y_knots]), extrapolate=True)(
                        np.log(freq_Hz[condition])
                        )
                residuals = skewnorm.rvs(alpha, loc=loc, scale=sigma,
                                         size=np.sum(condition))
                Y[condition, i] = spline + residuals

        # Add injections
        if fndr.injection_path is not None:
            beta = beta_H1 if ifo == "H1" else beta_L1
            injData = np.atleast_2d(utils.safe_loadtxt(fndr.injection_path, dtype=float))
            freq_amp_gen = zip(injData[:, 0], injData[:, 1])
            len_peak = len(fndr.peak_shape)
            for freq, amp in tqdm(freq_amp_gen, total=injData.shape[0],
                                    desc="Injecting DM signals", leave=False):
                # Check if frequency is in bound
                if freq <= fmin or freq >= fmax:
                    continue
                fmin_idx = max(0, sensutils.binary_search(freq_Hz, freq))
                fmax_idx = fmin_idx + len_peak
                if fmax_idx >= len(freq_Hz):
                    continue

                # Update peak shape and inject
                fndr.peak_shape.update_freq(freq)
                Y[fmin_idx:fmax_idx, i] =\
                    np.log(np.exp(Y[fmin_idx:fmax_idx, i]) +
                            beta[fmin_idx:fmax_idx]*amp*fndr.peak_shape)

    # # Compress idx_freq_to_df
    # def append_data(_row, _args, _knots, _rowval, _argval, _knotval):
    #     _row.append(_rowval)
    #     _args.append(_argval)
    #     _knots.append(_knotval)

    # compressed_idx_freq_to_df, compressed_args, compressed_knots = [], [], []
    # for j in range(idx_freq_to_df.shape[0]):  # N_segments
    #     compressed_row, _compressed_args, _compressed_knots = [], [], []
    #     current_val, count = None, 0

    #     for i, val in enumerate(idx_freq_to_df[j, :]):  # freqs
    #         if val != current_val:  # New df
    #             if i > 0:
    #                 append_data(compressed_row, _compressed_args, _compressed_knots,
    #                             [current_val, count], current_args, current_knots)
    #             current_val, count = val, 1
    #             current_knots, current_args = unique_knots[j, i, ...], unique_model_args[j, i, :]
    #         else:
    #             count += 1
    #     # Update current values and append to global idx data
    #     append_data(compressed_row, _compressed_args, _compressed_knots,
    #                 [current_val, count], current_args, current_knots)
    #     append_data(compressed_idx_freq_to_df, compressed_args, compressed_knots,
    #                 compressed_row, _compressed_args, _compressed_knots)
    return dict(Y=Y, beta_H1=beta_H1, beta_L1=beta_L1, ifos=ifos,
                fmin=freq_Hz[0], fmax=freq_Hz[-1],
                model_args=unique_model_args,
                knots=unique_knots)
                # model_args=np.array(compressed_args),
                # knots=np.array(compressed_knots),
                # idx_compression=np.array(compressed_idx_freq_to_df)
                # )


def create_job_args(prefix, fmin=10, fmax=5000, isMC=False, **kwargs):
    """Coordinate q0 analysis."""
    # Get data (MC or Real)
    fndr = DMFinder(**kwargs)
    if isinstance(fmin, int) or isinstance(fmax, int):
        fmin, fmax = [fmin], [fmax]

    for i, (_fmin, _fmax) in enumerate(tqdm(zip(fmin, fmax),
                                            desc="Filling args.", total=len(fmin))):
        args = make_args(fndr, _fmin, _fmax, isMC)
        np.savez(f"{prefix}_{i}.npz", **args)


if __name__ == '__main__':
    create_job_args(**parse_cmdl_args())
