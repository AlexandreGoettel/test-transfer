"""Communicate to and from disk for LPSD-relevant variables and results."""
import os
import glob
import csv
import json
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
# Project imports
from utils import LPSDVars


def get_A_star(tf_path):
    """Return dict of interpolant to the transfer functions for each ifo."""
    transfer_function = {}
    transfer_function["H1"] = pd.read_csv(os.path.join(tf_path, "Amp_Cal_LHO.txt"),
                                          delimiter="\t")
    transfer_function["L1"] = pd.read_csv(os.path.join(tf_path, "Amp_Cal_LLO.txt",
                                                       delimiter="\t"))
    f_A_star = {"H1": interp1d(transfer_function["H1"]["Freq_o"],
                               transfer_function["H1"]["Amp_Cal_LHO"]),
                "L1": interp1d(transfer_function["L1"]["Freq_Cal"],
                               transfer_function["L1"]["amp_cal_LLO"])}
    return f_A_star


def get_label(filepath=None, t0=None, t1=None, ifo=None, prefix="bkginfo"):
    """Get simple df label from filepath."""
    if filepath is None:
        assert all([x is not None for x in [t0, t1, ifo]])
        return f"{prefix}_{t0}_{t1}_{ifo}"
    else:
        main, _ = os.path.splitext(os.path.split(filepath)[-1])
        body = "_".join(main.split("_")[-3:])
        return f"{prefix}_{body}"


def sep_label(filepath):
    """Extract timestamp and ifo from LPSD output filepath."""
    t0, t1, ifo = os.path.splitext(os.path.split(filepath)[-1])[0].split("_")[-3:]
    return t0, t1, ifo


class LPSDJSONIO:
    """Read/Write from JSON files."""

    def __init__(self, filename):
        self.filename = filename
        self.data = None
        self.update_data()

    def update_data(self):
        with open(self.filename, "r") as _file:
            self.data = json.load(_file)

    def update_file(self, name, df, orient="records"):
        """Write data to the JSON file, give it reference "name"."""
        data = {}
        if os.path.exists(self.filename):
            with open(self.filename, "r") as _file:
                data = json.load(_file)

        # Update data with dataframe contents & save
        data[get_label(name)] = df.to_json(orient=orient)
        with open(self.filename, "w") as _file:
            json.dump(data, _file)
        self.update_data()

    def get_df(self, name):
        """Get a dataframe labelled "name" from the JSON file."""
        if not os.path.exists(self.filename):
            raise IOError(f"File '{self.filename}' does not exist..")

        with open(self.filename, 'r') as file:
            data = json.load(file)
        json_df = data.get(name)
        if json_df is None:
            raise IOError(f"'{name} is not in '{self.filename}..")

        # Convert the JSON object back to DataFrame
        return pd.read_json(json_df)


class LPSDDataGroup:
    """Gather & generalise several LPSD output files."""

    def __init__(self, data_path, data_prefix=""):
        """Read data found in data_path to numpy format."""
        self.data_path = data_path
        self.data_prefix = data_prefix
        self.buffer_path = self.create_buffer_path(data_path)

        if os.path.isdir(data_path):
            files = list(glob.glob(os.path.join(data_path, f"{data_prefix}*")))
        elif os.path.exists(data_path):
            files = [data_path]
        else:
            raise IOError("Invalid path: '{_data_path}'.")

        self.freq, self.logPSD, self.metadata = self.read_all_data(files)

    def create_buffer_path(self, data_path, suffix=".h5"):
        """Create path based on data_path to store interim HDF data for fast retrieval."""
        loc, body = os.path.split(data_path)
        (main, _) = os.path.splitext(body)
        return os.path.join(loc, f"buffer_{main}{suffix}")

    def read_all_data(self, files):
        """Read from all files in 'files' and combine."""
        files = sorted(files)  # For reproducibility

        if os.path.exists(self.buffer_path):
            print("[INFO] Buffer exists, ignoring raw data files..")
            with h5py.File(self.buffer_path, "r") as buffer:
                freq = np.array(buffer["freq"])
                logPSD = np.array(buffer["logPSD"])
                metadata = dict(buffer["logPSD"].attrs)
            metadata = self.convert_metadata_Nones(metadata)
            return freq, logPSD, metadata

        # Make sure that all files are compatible
        data = [LPSDOutput(f) for f in files]
        _len = len(data[0])
        assert not any([len(f) != _len for f in data[1:]])

        # Get metadata
        metadata = {"t0": [], "t1": [], "ifo": []}
        for x in data[0].kwargs.keys():
            metadata[x] = []
        for i, filename in enumerate(files):
            t0, t1, ifo = sep_label(filename)
            metadata["t0"].append(int(t0))
            metadata["t1"].append(int(t1))
            metadata["ifo"].append(ifo)
            for k, v in data[i].kwargs.items():
                metadata[k].append(v)

        # Combine to numpy array
        freq = data[0].freq
        output = np.zeros((len(data), _len), dtype=np.float64)
        for i, row in enumerate(data):
            output[i, :] = row.logPSD
        return freq, output, metadata

    def save_data(self):
        """Save gathered data to HDF format."""
        print(f"[INFO] Saving buffer to {self.buffer_path}..")
        self.metadata = self.convert_metadata_Nones(self.metadata)
        with h5py.File(self.buffer_path, "w") as outfile:
            outfile.create_dataset("freq", data=self.freq)
            outfile.create_dataset("logPSD", data=self.logPSD)
            outfile["logPSD"].attrs.update(self.metadata)

    def convert_metadata_Nones(self, metadata):
        """Convert -1 to None or viceversa."""
        for k, v in metadata.items():
            metadata[k] = [None if x == -1 else (-1 if x is None else x) for x in v]
        return metadata


class LPSDData:
    """Hold & manage LPSD output data."""

    def __init__(self, logPSD, freq=None, **kwargs):
        assert all([flag in kwargs for flag in ["fmin", "fmax", "fs", "Jdes"]])
        self.kwargs = kwargs
        self.logPSD = logPSD
        if freq is None:
            self.freq = self.freq_from_kwargs()
        # Get full LPSD vars
        if "resolution" not in kwargs:
            kwargs["resolution"] = np.exp((np.log(kwargs["fmax"]) - np.log(kwargs["fmin"])) /\
                (kwargs["Jdes"] - 1)) - 1.
        if "epsilon" not in kwargs:
            kwargs["epsilon"] = None
        self.vars = LPSDVars(*map(lambda x: kwargs[x],
                                  ["fmin", "fmax", "resolution", "fs", "epsilon"]))

    def __len__(self):
        return len(self.logPSD)

    def freq_from_kwargs(self):
        """Derive frequency from kwargs, can be more precise than reading from file."""
        return np.logspace(np.log10(self.kwargs["fmin"]),
                           np.log10(self.kwargs["fmax"]),
                           int(self.kwargs["Jdes"])
                           )


class LPSDOutput(LPSDData):
    """Extend LPSDData with functionality to read from an output file."""

    def __init__(self, filename):
        self.filename = filename
        is_hdf5 = filename.endswith(".h5") or filename.endswith(".hdf5")
        self.kwargs = self.get_lpsd_kwargs_hdf5() if is_hdf5 else self.get_lpsd_kwargs()
        # Check that file is valid
        assert self.kwargs
        # This function is only meant to process complete files
        if "batch" in self.kwargs:
            assert self.kwargs["Jdes"] == self.kwargs["batch"]

        # Get logPSD data from file
        if is_hdf5:
            self.freq, self.logPSD = self.read_hdf5()
        else:
            raw_freq, psd = self.read(raw_freq=True)
            # Protection against (old) LPSD bug
            self.logPSD = np.log(psd[:-1]) if psd[-1] == 0 else np.log(psd)
            self.freq = self.freq_from_kwargs() if len(self.logPSD) == int(self.kwargs["Jdes"])\
                else raw_freq

        super().__init__(self.logPSD, freq=self.freq, **self.kwargs)

    def get_lpsd_kwargs_hdf5(self, dset="logPSD"):
        """Extract LPSD parameters from a lpsd .h5 file."""
        with h5py.File(self.filename, "r") as _f:
            return dict(_f[dset].attrs)

    def get_lpsd_kwargs(self):
        """Extract LPSD parameters from an lpsd output file."""
        flags = {
            "fmax": "t",
            "fmin": "s",
            "fs": "f",
            "Jdes": "J",
            "batch": "n",
            "epsilon": "E"
        }
        with open(self.filename, "r", encoding="utf-8") as _f:
            for line in _f:
                if "Command line" in line:
                    line_flags = line.split()
                    values = [(float(line_flags[line_flags.index(f"-{flag}") + 1])
                            if f"-{flag}" in line_flags
                            else None)
                            for flag in flags.values()]
                    return dict(zip(flags.keys(), values))
        raise IOError("Invalid output file.")

    def read_hdf5(self, freq_dset="frequency", psd_dset="logPSD"):
        """Read freq_dset and psd_dset from datafile."""
        with h5py.File(self.filename, "r") as _f:
            x, y = _f[freq_dset][()], _f[psd_dset][()]
        return np.array(x), np.array(y)

    def read(self, dtype=np.float64, delimiter="\t", raw_freq=False):
        """
        Read an output file from LPSD.

        return: frequency & PSD arrays.
        """
        x, y = [], []
        with open(self.filename, "r") as _file:
            data = csv.reader(_file, delimiter=delimiter)
            for row in tqdm(data, total=int(self.kwargs["Jdes"]), desc="Reading LPSD", leave=False):
                try:
                    if raw_freq:
                        x += [float(row[0])]
                    y += [float(row[1])]
                except (ValueError, IndexError):
                    continue
        return np.array(x, dtype=dtype), np.array(y, dtype=dtype)
