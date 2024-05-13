"""Communicate to and from disk for LPSD-relevant variables and results."""
import os
import glob
import csv
import json
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
# Project imports
from utils import LPSDVars


class LPSDJSONIO:
    """Read/Write from JSON files."""

    def __init__(self, filename):
        self.filename = filename

    def update_file(self, name, df, orient="records"):
        """Write data to the JSON file, give it reference "name"."""
        data = {}
        if os.path.exists(self.filename):
            with open(self.filename, "r") as _file:
                data = json.load(_file)

        # Update data with dataframe contents & save
        data[name] = df.to_json(orient=orient)
        with open(self.filename, "w") as _file:
            json.dump(data, _file)

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

    def get_label(self, filepath, prefix="bkginfo"):
        """Get simple df label from filepath."""
        main, _ = os.path.splitext(os.path.split(filepath)[-1])
        body = "_".join(main.split("_")[-3:])
        return f"{prefix}_{body}"


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
                           )[:-1]


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
            self.freq = self.freq_from_kwargs()
            _, psd = self.read()
            # Protection against (old) LPSD bug
            self.logPSD = np.log(psd[:-1]) if psd[-1] == 0 else np.log(psd)

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
