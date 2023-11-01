"""Define classes to manage noise and signal data for injections."""
import os
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy import constants
import h5py
# Phil imports
from falsesignal import FalseSignal
# Project imports
import utils
import fft


BASE_PATH = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]


class DataManager:
    """Combine noise and signal data."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.nmax = kwargs["nmax"]

        self.datafile = h5py.File(self.kwargs["output_file"], "w")
        self.noise_generator = NoiseGenerator(self.datafile, **kwargs)
        self.signal_generator = SignalGenerator(self.datafile, **kwargs)

        # Initialise entropy
        np.random.seed()
        self.starting_seed = np.random.randint(int(2**31))

    def __del__(self):
        self.datafile.close()

    def add_injections(self, dname="PSD"):
        """Inject signals into the PSD in self.datafile[dname]."""
        # Use "injection_type", "injection_file", "injection_frequencies"
        injection_type = self.kwargs["injection_type"]
        if injection_type == "None":
            print("'injection-type' = None, skipping injections..")
            return
        # Get the injector for the kind of wave to inject
        if self.kwargs["wavetype"] == "DMlike":
            injector = self.signal_generator.inject_DM
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
        print("Injecting signals..")
        for freq, amp in tqdm(freq_amp_gen, total=N_gen,
                                desc="Injection frequency", position=1, leave=False):
            tqdm.write(f"AMP: {amp}")
            start_idx, end_idx, signal = injector(freq, amp, dname=dname)
            # Write to disk
            self.datafile[dname][start_idx:end_idx] += signal

    def generate_noise(self):
        """Generate data using the NoiseGenerator class."""
        self.noise_generator.generate_psd()

    def process_chunk(self, start, end, delta_f, dset, dset_index, seed=42):
        """Simulate complex-valued fft output noise in frequency domain."""
        M = int((end - start) / delta_f)
        freq_data = np.sqrt(2. * dset[dset_index:dset_index+M])  # PSD->ASD
        # Give each freq. a random phase and use it to generate a complex signal
        np.random.seed(self.starting_seed + seed)
        phase_data = np.random.random(size=len(freq_data)) * 2. * np.pi
        return freq_data * np.exp(1j * phase_data)

    def strain_from_PSD(self, dname="PSD"):
        """Generate strain data from PSD."""
        dset_PSD = self.datafile[dname]
        N = len(dset_PSD)
        delta_f, fs = dset_PSD.attrs["delta_f"], dset_PSD.attrs["sampling_frequency"]
        n_time = 2*N
        max_f = fs / 2.

        # Generate a temporary dataset to store phase
        dname_ASD = "ASD_complex"
        self.datafile.create_dataset(dname_ASD, (n_time,), dtype=np.complex128)
        dset = self.datafile[dname_ASD]

        # Loop over memory units to write quasi-symmetric f-array
        start, dset_index = delta_f, 0  # start in f-space, not index
        for start in tqdm(np.arange(delta_f, max_f + delta_f, self.nmax * delta_f),
                          desc="Create complex ASD"):
            # Loop condition
            end = min(max_f + delta_f, start + self.nmax * delta_f)

            # Create complex ASD data
            _seed = int((start - delta_f) / (self.nmax * delta_f)) + 1
            data = self.process_chunk(start, end, delta_f, dset_PSD, dset_index, _seed)

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
        if self.nmax <= n_time:
            dset_real = self.datafile.create_dataset("strain", (n_time,), dtype=np.float64)
            for start in trange(0, n_time, self.nmax,
                                desc="Write real strain", position=2, leave=False):
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

    def psd_from_spline(self):
        """Generate a function from pre-fitted spline data."""
        # Spline is defined between 10 and 8192 Hz
        # H1:
        # xKnots = np.array([2.30258509, 3.04941017, 3.79623525,
        #                    8.65059825, 8.95399594, 8.97733423, 9.02401079])
        # yKnots = np.array([-91.80878694876485, -99.97801940114547, -103.57729069085298,
        #                    -102.17121965438, -104.34025547329, -105.9256036217, -130.06995841416])
        # L1:
        xKnots = np.array([2.30258509, 3.04941017, 3.79623525, 8.65059825,
                           8.95399594, 8.97733423, 9.02401079])
        yKnots = np.array([-91.64265596, -97.29276278, -101.0729336, -101.77673911,
                           -105.56706334, -106.59332013, -107.01675492])
        def linspace_spline(x):
            # This is needed because the spline was fitted in log-log space
            return np.exp(CubicSpline(xKnots, yKnots, extrapolate=False)(np.log(x)))
        return linspace_spline

    def psd_func(self, N):
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

            # Normalise
            return x*(N)**2 / 16
        return extrapolator

    def generate_psd(self, dname="PSD"):
        """Generate the PSD and write it to disk."""
        # Define variables
        fs, T = self.kwargs["sampling_frequency"], self.kwargs["length"]
        # Adjust T so that n_time is a power of two
        n_time = fft.smallest_power_of_two_above(2*int(T*fs / 2.))
        if n_time > 2*int(T*fs / 2.):
            print(f"Warning: writing t={n_time/fs:.1f}s instead of t={T}s"+
                  " of strain to satisfy Nfft condition.")
        delta_f = fs / n_time
        max_f = fs / 2.  # Hz

        # Create h5 file to store complex data out of memory
        self.datafile.create_dataset(dname, (n_time//2,), dtype=np.float64)
        dset = self.datafile[dname]

        # Loop over memory units to write quasi-symmetric f-array
        psd_func = self.psd_func(n_time)
        dset_index = 0
        for start in tqdm(np.arange(delta_f, max_f + delta_f, self.nmax * delta_f),
                          desc="Generate noise"):
            # Loop condition
            end = min(max_f + delta_f, start + self.nmax * delta_f)

            # Create complex ASD data
            positive_freqs = np.arange(start, end, delta_f)
            M = len(positive_freqs)
            dset[dset_index:dset_index + M] = psd_func(positive_freqs)

            dset_index += M

        # Set attributes for readability
        dset.attrs["sampling_frequency"] = self.kwargs["sampling_frequency"]
        dset.attrs["f0"] = delta_f
        dset.attrs["delta_f"] = delta_f


class SignalGenerator:
    """Create time domain injections for sine-like and DM-like signals."""
    def __init__(self, datafile, sampling_frequency=16384., **kwargs):
        """Init fs/datafile and prep. DM conversion assuming H1."""
        self.resolution = kwargs["resolution"]
        self.fs = sampling_frequency
        self.datafile = datafile
        # Open transfer function file for DM-amp conversion
        tf_dir = os.path.join(BASE_PATH, "scripts", "data", "transfer_functions")
        tf_data = pd.read_csv(os.path.join(tf_dir, "Amp_Cal_LLO.txt"), delimiter="\t")
        f_A_star = interp1d(tf_data["Freq_Cal"], tf_data["amp_cal_LLO"])

        # Calculate beta
        rho_local = 0.4 / (constants.hbar / constants.e * 1e-7 * constants.c)**3  # to GeV^4
        self.beta = lambda f_Hz: rho_local / (np.pi*f_Hz**3 * f_A_star(f_Hz)**2
                                              * (constants.e / constants.h)**2)

    def inject_sine(self, f, A, dname="PSD"):
        """Inject a sine wave into the data."""
        # Find correct frequency bin
        dset = self.datafile[dname]
        N, f0, delta_f = len(dset), dset.attrs["f0"], dset.attrs["delta_f"]

        idx = int(np.ceil((f - f0) / delta_f))
        amplitude = (A * N)**2 / 8
        return idx, idx+1, np.ones(1) * amplitude

    # def inject_DM(self, start_idx, end_idx, f, A):
    #     """See inject_sine but mimic DM-like signal based on Phil's code."""
    #     # The DM signal is the overlap of many sine waves
    #     signal = FalseSignal(
    #         frequency=f,
    #         amplitude=A,
    #         phase_seed=np.random.randint(2**31),
    #         Nfreqs=500,
    #         FWHM=1e-6,
    #         day=np.random.randint(365)
    #     )
    #     output = np.zeros(end_idx - start_idx)
    #     t = np.arange(start_idx, end_idx) / self.fs
    #     for A, f, phase in tqdm(zip(signal["amplitudes"], signal["frequencies"], signal["phases"]),
    #                             desc="DM freqs", total=500):
    #         output += A * np.sin(2.*np.pi * f * t + phase)

    #     return output
    # TODO INJECT_DM_FROM_FREQ!
    def DM_line_shape(self, freq, remainder=1e-3):
        """
        Return interpolant to DM line shape, and bounds.

        Determine bounds by numerically integrating up to remainder.
        """
        tau = 1. / (freq * self.resolution)
        xmin = freq - 1. / (2*tau)
        xmax = freq + 10. / tau
        def analytical_line_shape(x, tau, x0):
            return tau/np.sqrt(2*np.pi)*np.exp(-tau*(x-x0)-1) * np.sinh(np.sqrt(1 + 2*tau*(x-x0)))

        def to_solve(x, xmin, tau, freq, alpha):
            if x < xmin:
                return 100
            val = quad(analytical_line_shape, xmin, x, args=(tau, freq))[0]
            return (val - 0.5*(1-alpha))**2

        upper_bound_solved = fsolve(to_solve, xmax, args=(xmin, tau, freq, remainder))

        N = 1000
        x = np.linspace(xmin, upper_bound_solved, 1000).reshape((N,))
        y = analytical_line_shape(x, tau, freq).reshape((N,))
        return xmin, upper_bound_solved, interp1d(x, y)

    def inject_DM(self, f, A, dname="PSD"):
        """Inject a DM-like signal in frequency space."""
        # Get frequency axis info
        dset = self.datafile[dname]
        N, f0, delta_f = len(dset), dset.attrs["f0"], dset.attrs["delta_f"]

        # Get DM line shape and convert for practical use
        fmin, fmax, f_line_shape = self.DM_line_shape(f)
        # idx = int(np.ceil((f - f0) / delta_f))
        idx_xmin = int(np.ceil((fmin - f0) / delta_f))
        idx_xmax = int(np.ceil((fmax - f0) / delta_f))
        freqs = np.arange(fmin, fmax, delta_f)
        line_shape = f_line_shape(freqs)

        # Normalise and return
        PSD_factor = N**2/8
        norm = self.beta(f)*A**2 * PSD_factor
        return idx_xmin, idx_xmax, norm*line_shape[:idx_xmax - idx_xmin]
