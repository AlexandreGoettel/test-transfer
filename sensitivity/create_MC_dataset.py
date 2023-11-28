"""Create full 40-segment PSD MC dataset through direct injection."""
import os
import sys
import glob
from tqdm import tqdm, trange
import h5py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, skewnorm
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy import constants


sys.path.append("../scripts/")
import utils
import sensutils
import models


BASE_PATH = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]


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


class MCGenerator:
    """Generate MC data."""

    def __init__(self, data_path, json_path, injection_path, output_path, peak_shape_path,
                 dname="MC_injected", _MAX_CHI=10, seed=0,
                 resolution=1e-6, sampling_frequency=16384):
        # Set variables
        self.data_path = data_path
        self.json_path = json_path
        self.injection_path = injection_path
        self.output_path = output_path
        self.peak_shape = PeakShape(0, peak_shape_path)
        self.dname = dname
        self._MAX_CHI = _MAX_CHI
        self.resolution = resolution
        self.sampling_frequency = sampling_frequency
        self.seed = seed

        # Init HDF data
        self.data_info = None
        self.dfile = None
        self.freq = None

    def __del__(self):
        """Make sure HDF files are closed properly."""
        if hasattr(self, "data_info"):
            for [dfile, *_] in self.data_info:
                dfile.close()
        if hasattr(self, "dfile") and self.dfile:
            self.dfile.close()

    def open_HDFs(self):
        """Open all relevant data HDFs."""
        self.data_info = self.get_info_from_data_path()

    def do_data_to_dfile(self):
        """Read PSD data and write relevant bits to dfile."""
        self.dfile = self.fill_dset()

    def get_info_from_data_path(self):
        """Get all possible info from result files in data_path and json_path."""
        data_paths = sorted(list(glob.glob(os.path.join(self.data_path, "result*"))))
        data_info = []
        for _data_path in tqdm(data_paths, desc="Opening HDFs", leave=False):
            _data_info = []
            _data_info.append(h5py.File(sensutils.get_corrected_path(_data_path), "r"))
            ifo = sensutils.parse_ifo(_data_path)
            _data_info.append(ifo)
            df_key = "splines_" + sensutils.get_df_key(_data_path)
            _data_info.append(sensutils.get_results(df_key, self.json_path))
            assert _data_info[-1] is not None
            _data_info.append(df_key)
            data_info.append(_data_info)

        return data_info

    def get_frequencies(self):
        """Get frequency array from data_info."""
        assert self.data_info
        psd_lengths = [len(dat[0]["frequency"]) for dat in self.data_info]
        return self.data_info[np.argmax(psd_lengths)][0]["frequency"][:]

    def get_beta(self, ifo):
        """Return ifo-specific transfer function."""
        # Open transfer function file for DM-amp conversion
        tf_dir = os.path.join(BASE_PATH, "scripts", "data", "transfer_functions")
        if ifo == "L1":
            tf_data = pd.read_csv(os.path.join(tf_dir, "Amp_Cal_LLO.txt"), delimiter="\t")
            f_A_star = interp1d(tf_data["Freq_Cal"], tf_data["amp_cal_LLO"])
        else:
            tf_data = pd.read_csv(os.path.join(tf_dir, "Amp_Cal_LHO.txt"), delimiter="\t")
            f_A_star = interp1d(tf_data["Freq_o"], tf_data["Amp_Cal_LHO"])

        # Calculate beta
        rho_local = 0.4 / (constants.hbar / constants.e * 1e-7 * constants.c)**3  # to GeV^4
        def beta(f_Hz):
            return rho_local / (np.pi*f_Hz**3 * f_A_star(f_Hz)**2
                                * (constants.e / constants.h)**2)
        return beta

    def fill_dset(self):
        """Create a dataset to hold all of the PSDs."""
        # Create dataset with high frequency tolerance to be cropped later
        psd_lengths = [len(dat[0]["frequency"]) for dat in self.data_info]
        psd_length = max(psd_lengths)
        dfile = h5py.File(self.output_path, "a")
        dset = dfile.create_dataset(self.dname, (psd_length, len(self.data_info)),
                                    dtype=np.float64)

        fillBkg = False
        if "bkg" not in dfile.keys():
            bkg = dfile.create_dataset("bkg", (psd_length, len(self.data_info)),
                                       dtype=np.float64)
            fillBkg = True

        fillModelArgs = False
        if "model_args" not in dfile.keys():
            model_args = dfile.create_dataset("model_args", (psd_length, len(self.data_info), 3),
                                              dtype=np.float64)
            fillModelArgs = True

        isH1 = dfile.create_dataset("isH1", shape=(dset.shape[1]), dtype=bool)
        # Metadata
        dset.attrs["json_file"] = self.json_path

        # Fill the dset
        df_columns = ["fmin", "fmax", "x_knots", "y_knots",
                      "alpha_skew", "loc_skew", "sigma_skew", "chi_sqr"]
        np.random.seed(self.seed)
        for i, (hdf_path, ifo, df, df_key) in enumerate(
                tqdm(self.data_info, desc="Fill dset.", leave=False)):
            for _, row in df.iterrows():
                fmin, fmax, x_knots, y_knots, alpha_skew, loc_skew, sigma_skew, chi_sqr\
                    = row[df_columns]
                if chi_sqr >= self._MAX_CHI:
                    continue

                # Get frequency indices
                frequencies = hdf_path["frequency"]
                idx_min = sensutils.binary_search(frequencies, fmin)
                idx_max = sensutils.binary_search(frequencies, fmax)

                # Construct spline base
                spline = models.model_xy_spline(
                    np.concatenate([x_knots, y_knots]), extrapolate=True)(
                        np.log(frequencies[idx_min:idx_max+1]))

                # Add randomised residuals
                residuals = skewnorm.rvs(alpha_skew, loc=loc_skew, scale=sigma_skew,
                                         size=idx_max+1-idx_min)
                dset[idx_min:idx_max+1, i] = spline + residuals
                if fillBkg:
                    bkg[idx_min:idx_max+1, i] = spline
                if fillModelArgs:
                    # Is there a faster way?
                    for j, val in enumerate([alpha_skew, loc_skew, sigma_skew]):
                        model_args[idx_min:idx_max+1, i, j] = val
            # Save metadata
            dset.attrs[str(i)] = df_key
            isH1[i] = ifo == "H1"
        # Clean dataset (loop to avoid excessive memory usage)
        for j in trange(dset.shape[1], desc="Cleaning"):
            dset[dset[:, j] == 0, j] = np.nan

        # Write frequencies to output dfile
        if "frequencies" not in dfile.keys():
            dfile["frequencies"] = self.get_frequencies()

        return dfile

    def add_injections(self):
        """Inject DM-like peaks."""
        beta_H1, beta_L1 = self.get_beta("H1"), self.get_beta("L1")
        frequencies = self.get_frequencies()
        isH1 = self.dfile["isH1"][:]
        dset = self.dfile[self.dname]
        injData = np.atleast_2d(utils.safe_loadtxt(self.injection_path, dtype=float))
        freq_amp_gen = zip(injData[:, 0], injData[:, 1])
        len_peak = len(self.peak_shape)
        for freq, amp in tqdm(freq_amp_gen, total=injData.shape[0],
                              desc="Injecting DM signals", leave=False):
            # Set peak variables
            A_H1 = beta_H1(freq) * amp**2
            A_L1 = beta_L1(freq) * amp**2
            fmin_idx = max(0, sensutils.binary_search(frequencies, freq) - len_peak//2)
            fmax_idx = fmin_idx + len_peak

            # Apply
            self.peak_shape.update_freq(freq)
            # Renorm shouldn't be necessary but doesn't hurt!
            signal = self.peak_shape.reshape((len_peak, 1)) / np.sum(self.peak_shape)

            if fmax_idx > dset.shape[0]:
                tqdm.write(f"Skipping {freq}, too close to boundary..")
                continue
            dset[fmin_idx:fmax_idx, isH1] =\
                np.log(np.exp(dset[fmin_idx:fmax_idx, isH1]) + A_H1*signal)
            dset[fmin_idx:fmax_idx, ~isH1] =\
                np.log(np.exp(dset[fmin_idx:fmax_idx, ~isH1]) + A_L1*signal)


def main(data_path, json_path, injection_path, output_path, peak_shape_path,
         skip_injection=False, **kwargs):
    """Manage script."""
    # Initialise
    generator = MCGenerator(data_path, json_path, injection_path,
                            output_path, peak_shape_path, **kwargs)
    generator.open_HDFs()
    generator.do_data_to_dfile()

    if not skip_injection:
        generator.add_injections()
    del generator.dfile["isH1"]

    # Representative plot - sum of all PSDs
    dset = generator.dfile[generator.dname]
    frequencies = generator.get_frequencies()

    plt.figure()
    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _sum = np.zeros(dset.shape[0])
    for j in trange(dset.shape[1], desc="Add results for plot"):
        _sum += np.exp(dset[:, j])
    ax.plot(frequencies, _sum)

    plt.show()


if __name__ == '__main__':
    main("../scripts/data/",
         "../scripts/data/processing_results.json",
         "data/injections/injections_full_1.0e-17.dat",
         "MC.h5",
         "../scripts/peak_shape_data.npz",
        #  dname="noise",
         dname="injection_1e-17",
         seed=42,
         skip_injection=False
         )
