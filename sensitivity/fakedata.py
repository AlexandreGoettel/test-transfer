"""Define classes to manage noise and signal data for injections."""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import h5py
import utils


class DataManager:
    """Combine noise and signal data."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.noise_generator = NoiseGenerator(**kwargs)
        self.signal_generator = SignalGenerator(**kwargs)
        self.time_data, self.freq_data = None, None

    def add_injections(self):
        """Inject signals into the time series."""
        # Use "injection_type", "injection_file", "injection_frequencies"
        injection_type = self.kwargs["injection_type"]
        if injection_type == "None":
            print("injection-type = None, skipping injections..")
            return
        if self.kwargs["wavetype"] != "sinelike":
            raise ValueError("Only sinelike injections implemented.")  # TODO
        else:
            injector = self.signal_generator.inject_sine

        if injection_type == "given-frequencies":
            for freq, amp in zip(self.kwargs["injection_frequencies"],
                                 self.kwargs["injection_amplitudes"]):
                self.time_data["strain"] += injector(len(self.time_data["strain"]), freq, amp)

        if injection_type == "injection-file":
            injData = utils.safe_loadtxt(self.kwargs["injection_file"], dtype=float)
            for freq, amp in zip(injData[:, 0], injData[:, 1]):
                self.time_data["strain"] += injector(len(self.time_data["strain"]), freq, amp)

    def save_data(self):
        """Save PSD and time series to HDF."""
        # Check if passed output filename is valid
        filename = self.kwargs["output_file"]
        assert filename.endswith(".h5") or filename.endswith(".hdf5")

        # Save strain and PSD as separate datasets
        print(f"Writing to {filename}..")
        with h5py.File(filename, 'w') as hdf_file:
            hdf_file.create_dataset("strain", data=self.time_data["strain"].values)
            hdf_file.create_dataset("PSD", data=self.freq_data["PSD"].values)

        # Add attributes for clarity
        with h5py.File(filename, "a") as _f:
            _f["strain"].attrs["sampling_frequency"] = self.kwargs["sampling_frequency"]
            f0, f1 = self.freq_data["frequencies"][0], self.freq_data["frequencies"][1]
            _f["PSD"].attrs["f0"] = f0
            _f["PSD"].attrs["delta_f"] = f1 - f0

    def generate_noise(self):
        """Generate data using the NoiseGenerator class."""
        self.time_data, self.freq_data = self.noise_generator.freq_to_time()

    def plot_data(self):
        """Plot the strain and ASD data if it exists."""
        if self.time_data is not None and "strain" in self.time_data:
            plt.figure()
            plt.plot(self.time_data["time"], self.time_data["strain"])
            plt.xlabel("Time (s)")
            plt.ylabel("Strain")

        if self.freq_data is not None and "ASD" in self.freq_data:
            plt.figure()
            ax = plt.subplot(111)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.plot(self.freq_data["frequencies"], self.freq_data["PSD"])
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD")

        plt.show()


class NoiseGenerator:
    """Create time domain noise from PSDs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def psd_from_spline(self):
        """Generate a function from pre-fitted spline data."""
        # Spline is defined between 10 and 8192 Hz
        xKnots = np.array([2.30258509, 3.04941017, 3.79623525,
                           8.65059825, 8.95399594, 8.97733423, 9.02401079])
        yKnots = np.array([-91.80878694876485, -99.97801940114547, -103.57729069085298,
                           -102.17121965438, -104.34025547329, -105.9256036217, -130.06995841416])

        def linspace_spline(x):
            # This is needed because the spline was fitted in log-log space
            return np.exp(CubicSpline(xKnots, yKnots, extrapolate=False)(np.log(x)))
        return linspace_spline

    def psd_func(self):
        """
        Define the value of the PSD for any frequency using extrapolation.
        
        Set time, strain, frequencies, ASD
        """
        def extrapolator(X):
            if self.kwargs["noise_source"] == "spline":
                x = self.psd_from_spline()(X)
            else:
                pass  # TODO get time series from data file directly

            # Don't return NaN outside of the PSD range but first and last non-NaN values instead
            not_nan = np.where(~np.isnan(x))
            first_non_nan = not_nan[0][0]
            last_non_nan = not_nan[0][-1]
            if first_non_nan:
                x[:first_non_nan] = x[first_non_nan]
            if last_non_nan - len(x):
                x[last_non_nan+1:] = x[last_non_nan]

            return x
        return extrapolator

    def freq_to_time(self):
        """Generate a random time series from the PSD."""
        delta_f = 1. / (2.*self.kwargs["length"])
        max_f = self.kwargs["sampling_frequency"] / 2.  # Hz

        # Simulate complex-valued fft output noise in frequency domain
        positive_freqs = np.arange(delta_f, max_f+delta_f, delta_f)
        if len(positive_freqs) % 2 == 0:
            positive_freqs = np.append(positive_freqs, [positive_freqs[-1]+delta_f])

        # Give each freq. a random phase and use it to generate a complex signal
        freq_data = np.sqrt(2.*self.psd_func()(positive_freqs))  # ASD
        N = len(freq_data)
        phase_data = np.random.random(size=N)*2.*np.pi
        complex_freq_data = freq_data * np.exp(1j*phase_data)

        # Inverse fft to get time domain data
        data = np.zeros(2*N + 1, dtype=complex)
        data[0] = complex_freq_data[-1]
        data[1:N+1] = complex_freq_data
        data[N+1:] = np.conjugate(complex_freq_data[::-1])

        # Save output
        time_data = pd.DataFrame(
            {"strain": np.fft.ifft(data)[:N-1].real,
             "time": np.arange(0, 1./(2. * max_f)*(N-1), 1./(2. * max_f))}
            )
        freq_data = pd.DataFrame(
             {"frequencies": positive_freqs,
             "ASD": complex_freq_data,
             "PSD": .5*(complex_freq_data.real**2 + complex_freq_data.imag**2)}
             )
        return time_data, freq_data


class SignalGenerator:
    """Create time domain injections for sine-like and DM-like signals."""
    def __init__(self, sampling_frequency=16384., **_):
        self.fs = sampling_frequency

    def inject_sine(self, N, f, A):
        """Inject a sine wave of frequency f and amplitude A in a N-length array."""
        t = np.arange(N) / self.fs
        return A * np.sin(2.*np.pi * f * t)
