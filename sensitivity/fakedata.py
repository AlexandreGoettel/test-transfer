"""Define classes to manage noise and signal data for injections."""
from tqdm import tqdm, trange
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import h5py
# Project imports
import utils
import fft


class DataManager:
    """Combine noise and signal data."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.nmax = kwargs["nmax"]

        self.datafile = h5py.File(self.kwargs["output_file"], "w")
        self.noise_generator = NoiseGenerator(self.datafile, **kwargs)
        self.signal_generator = SignalGenerator(self.datafile, **kwargs)

    def __del__(self):
        self.datafile.close()

    def add_injections(self, dset="strain"):
        """Inject signals into the time series in self.datafile[dset]."""
        # Use "injection_type", "injection_file", "injection_frequencies"
        injection_type = self.kwargs["injection_type"]
        if injection_type == "None":
            print("'injection-type' = None, skipping injections..")
            return
        # Get the injector for the kind of wave to inject
        if self.kwargs["wavetype"] != "sinelike":
            raise ValueError("Only sinelike injections implemented.")  # TODO
        else:
            injector = self.signal_generator.inject_sine

        # Get the frequencies and amplitudes to inject
        if injection_type == "given-frequencies":
            freq_amp_gen = zip(self.kwargs["injection_frequencies"],
                               self.kwargs["injection_amplitudes"])
            N_gen = len(self.kwargs["injection_frequencies"])

        elif injection_type == "injection-file":
            injData = utils.safe_loadtxt(self.kwargs["injection_file"], dtype=float)
            injData = np.atleast_2d(injData)
            freq_amp_gen = zip(injData[:, 0], injData[:, 1])
            N_gen = injData.shape[0]

        # Perform injection(s)
        N = len(self.datafile[dset])
        print("Injecting signals..")
        for start_index in trange(0, N, self.nmax,
                                  desc="Memory unit", position=0):
            end_index = min(start_index + self.nmax, N)
            signal = np.zeros(end_index - start_index)
            for freq, amp in tqdm(freq_amp_gen, total=N_gen,
                                  desc="Injection frequency", position=1, leave=False):
                signal += injector(start_index, end_index, freq, amp)
            # Write to file
            self.datafile[dset][start_index:end_index] += signal

    def generate_noise(self):
        """Generate data using the NoiseGenerator class."""
        self.noise_generator.freq_to_time()

    def plot_data(self):
        """Plot the strain and ASD data if it exists."""
        if "strain" in self.datafile:
            # Check if there is more data than nmax, reduce appropriately
            dset = self.datafile["strain"]
            stride = max(1, int(1 + len(dset) / self.nmax))
            strain = dset[::stride]
            t = np.arange(0, len(dset)*dset.attrs["delta_t"],
                          dset.attrs["delta_t"]*stride)

            plt.figure()
            plt.plot(t, strain)
            plt.xlabel("Time (s)")
            plt.ylabel("Strain")

        if "ASD_complex" in self.datafile:
            # Idem for PSD
            dset = self.datafile["ASD_complex"]
            stride = max(1, int(1 + len(dset) / self.nmax))
            N = (len(dset) - 1) // 2
            ASD = dset[1:N+1:stride]
            PSD = .5 * (ASD.real**2 + ASD.imag**2)
            freq = np.arange(dset.attrs["f0"],
                             dset.attrs["delta_f"]*len(PSD) + dset.attrs["f0"],
                             dset.attrs["delta_f"])
            plt.figure()
            ax = plt.subplot(111)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.plot(freq, PSD)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD")

        plt.show()


class NoiseGenerator:
    """Create time domain noise from PSDs."""

    def __init__(self, datafile, **kwargs):
        self.kwargs = kwargs
        self.nmax = kwargs["nmax"]  # How much to put in memory at one time
        self.datafile = datafile
        # Initialise entropy
        np.random.seed()
        self.starting_seed = np.random.randint(int(2**31))

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

    def process_chunk(self, start, end, delta_f, psd_func, seed=42):
        """Simulate complex-valued fft output noise in frequency domain."""
        positive_freqs = np.arange(start, end, delta_f)
        freq_data = np.sqrt(2. * psd_func(positive_freqs))  # PSD->ASD
        # Give each freq. a random phase and use it to generate a complex signal
        np.random.seed(self.starting_seed + seed)
        phase_data = np.random.random(size=len(freq_data)) * 2. * np.pi
        return freq_data * np.exp(1j * phase_data)

    def freq_to_time(self):
        """Generate a random time series from the PSD."""
        # Define variables
        fs, T = self.kwargs["sampling_frequency"], self.kwargs["length"]
        # Adjust T so that n_time is a power of two
        n_time = fft.smallest_power_of_two_above(2*int(T*fs / 2.))
        if n_time > 2*int(T*fs / 2.):
            print(f"Warning: writing t={n_time/fs:.1f}s instead of t={T}s"+
                  " of strain to satisfy Nfft condition.")
        T = n_time / fs
        delta_f = fs / n_time
        max_f = self.kwargs["sampling_frequency"] / 2.  # Hz

        # Create h5 file to store complex data out of memory
        dname = "ASD_complex"
        self.datafile.create_dataset(dname, (n_time,), dtype=np.complex128)
        dset = self.datafile[dname]

        # Loop over memory units to write quasi-symmetric f-array
        start, dset_index = delta_f, 0  # start in f-space, not index
        psd_func = self.psd_func()
        print("Generating noise..")
        for start in tqdm(np.arange(delta_f, max_f + delta_f, self.nmax * delta_f)):
            # Loop condition
            end = min(max_f + delta_f, start + self.nmax * delta_f)

            # Create complex ASD data
            _seed = int((start - delta_f) / (self.nmax * delta_f)) + 1
            data = self.process_chunk(start, end, delta_f, psd_func, _seed)

            # Write in fft-output format to prepare for iFFT
            M = len(data)
            dset[1+dset_index:1+dset_index + M] = data

            last_iteration = start + self.nmax * delta_f >= max_f + delta_f
            if dset_index:
                dset[-(dset_index+M)+int(last_iteration):-dset_index] = np.conjugate(data[-2::-1])\
                    if last_iteration else np.conjugate(data[::-1])
            else:
                dset[-(dset_index+M)+int(last_iteration):] = np.conjugate(data[-2::-1])\
                    if last_iteration else np.conjugate(data[::-1])
            dset_index += M
        dset[0] = 0  # DC-component

        # Aply iFFT and store "complex_strain" dataset
        self.datafile.create_dataset("complex_strain", (n_time,), dtype=np.complex128)
        if n_time <= self.nmax:
            print("Performing NORMAL iFFT..")
            self.datafile["complex_strain"][:] = fft.FFT(dset, reverse=True, is_top_level=True)
        else:
            print("Performing MEMORY iFFT..")
            fft.memory_FFT(n_time, n_time, self.nmax, self.datafile, self.datafile,
                           dname, "complex_strain", reverse=True)
            del self.datafile[dname]

        # Save only real part
        # Imaginary parts should be negligible anyway
        dset_strain = self.datafile["complex_strain"]
        print("Project complex to real..")
        if self.nmax <= n_time:
            dset_real = self.datafile.create_dataset("strain", (n_time,), dtype=np.float64)
            for start in range(0, n_time, self.nmax):
                end = min(start + self.nmax, n_time)
                dset_real[start:end] = dset_strain[start:end].real
        else:
            # If n_time is not greater than nmax, just replace it in one go
            dset_new = dset_strain[:n_time].real
            del self.datafile["complex_strain"]
            self.datafile.create_dataset("strain", data=dset_new, dtype=np.float64)

        # Set attributes for readability
        dset.attrs["sampling_frequency"] = self.kwargs["sampling_frequency"]
        dset.attrs["f0"] = delta_f
        dset.attrs["delta_f"] = delta_f
        self.datafile["strain"].attrs["delta_t"] = 1. / self.kwargs["sampling_frequency"]


class SignalGenerator:
    """Create time domain injections for sine-like and DM-like signals."""
    def __init__(self, datafile, sampling_frequency=16384., **_):
        self.fs = sampling_frequency
        self.datafile = datafile

    def inject_sine(self, start_idx, end_idx, f, A):
        """Inject a sine wave of frequency f and amplitude A in a N-length array."""
        t = np.arange(start_idx, end_idx) / self.fs
        return A * np.sin(2. * np.pi * f * t)
