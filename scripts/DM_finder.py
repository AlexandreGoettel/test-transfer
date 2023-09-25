"""
Run on block-corrected data similarly to calc_upper_limit.

Could create data manager class?
Log-likelihood used to calculate q0 instead of q_mu
Idea here: Use f(q0|0) and q0 from data and that's it ^_^
Have to compare f(q0|0) between MC and theory (sigma-independent!)
"""
import os
import argparse
from multiprocessing import Pool
import glob
from tqdm import tqdm
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import constants
from scipy.optimize import minimize
from scipy.stats import norm, chi2
from scipy.interpolate import interp1d
# Project imports
import sensutils
import models


# tmp
import warnings
warnings.filterwarnings("ignore")


def log_likelihood(params, Y, bkg, peak_norm, model_args,
                   calib_args=None, doCalib=False):
    """Likelihood of finding dark matter in the data."""
    # Actual likelihood calculation
    if doCalib:
        mu_DM, theta = params
        residuals = 2*np.log(theta) + Y - np.log(np.exp(bkg) + peak_norm*mu_DM)
        gaussian_pull_term = sensutils.logpdf_skewnorm(  # skew-norm with zero skew is normal!
            theta, 0, loc=calib_args[:, 0], scale=calib_args[:, 1])
    else:
        mu_DM = params
        residuals = Y - np.log(np.exp(bkg) + peak_norm*mu_DM)
        gaussian_pull_term = 0

    skewnorm_term = sensutils.logpdf_skewnorm(
        residuals, model_args[:, 0], model_args[:, 1], model_args[:, 2])
    lkl = skewnorm_term + gaussian_pull_term
    # Security
    if len(lkl.shape) == 2:
        return np.sum(lkl, axis=1)
    elif len(lkl.shape) == 1:
        return lkl.sum()
    else:
        raise ValueError


def get_discovery_significance(Y, bkg, peak_norm, model_args, calib_args, do_calib=False):
    """Get discovery significance from f(q0|0)."""
    # 1 Fast answer
    #  1.1 Maximise lkl
    if do_calib:
        def log_lkl(params):
            return log_likelihood(params, Y, bkg, peak_norm, model_args,
                                  calib_args, doCalib=True)
    else:
        def log_lkl(params):
            return log_likelihood(params[0], Y, bkg, peak_norm, model_args)

    popt = minimize(lambda x: -log_lkl(x), 1e-36,
                    method="Nelder-Mead", tol=1e-10)
    max_lkl = -popt.fun

    #  1.2 return sqrt(q0|data)
    q_0 = -2*(log_lkl([0]) - max_lkl) if popt.x > 0 else 0
    return np.sqrt(q_0)
    # 2 Full answer
    #  2.1 For each simulation
    #  2.2 Maximum lkl and get Z from p-value
    #  2.3 Compare curve to formula


def process_segment(args):
    """Calculate an upper limit - wrapper for multiprocessing."""
    i, freq_Hz, do_calib, Y, bkg, model_args, peak_norm, calib_args = args
    kwargs = {"do_calib": do_calib}
    return i, freq_Hz, get_discovery_significance(Y, bkg, peak_norm, model_args,
                                                  calib_args, **kwargs)


def main(data_path, n_frequencies=2000, n_processes=4,
         max_chi_sqr=10, do_calib=False, json_path="data/processing_results.json"):
    """Get all necessary data and launch analysis."""
    # Analysis constants
    df_columns = ["x_knots", "y_knots", "alpha_skew", "loc_skew", "sigma_skew", "chi_sqr"]
    rho_local = 0.4 / (constants.hbar / constants.e * 1e-7 * constants.c)**3  # GeV/cm^3 to GeV^4

    # Paths
    data_paths = list(glob.glob(os.path.join(data_path, "result*")))
    if not data_paths:
        raise ValueError(f"No valid data found in '{data_path}'!")
    # TODO: rel. paths
    transfer_function_path = "../shared_git_data/Calibration_factor_A_star.txt"
    calib_dir = "calibration"

    # 0. Get A_star
    tf = np.loadtxt(transfer_function_path, delimiter="\t")
    f_A_star = {"H1": interp1d(tf[:, 1], tf[:, 0]),
                "L1": interp1d(tf[:, 1], tf[:, 0])}  # TODO: other file
    # 0.1 Open O3a and O3b calibration envelope files
    calib_gps_times = {
        "H1": [int(f.split("_")[4]) for f in glob.glob(os.path.join(
            calib_dir, "H1", "*FinalResults.txt"))],
        "L1": [int(f.split("_")[4]) for f in glob.glob(os.path.join(
            calib_dir, "L1", "*FinalResults.txt"))]
    }
    # 0.2 Open data HDFs
    data_info = []
    for _data_path in tqdm(data_paths, desc="Opening HDFs", leave=False):
        _data_info = []
        _data_info.append(h5py.File(sensutils.get_corrected_path(_data_path), "r"))
        ifo = sensutils.parse_ifo(_data_path)
        _data_info.append(ifo)
        df_key = "splines_" + sensutils.get_df_key(_data_path)
        _data_info.append(sensutils.get_results(df_key, json_path))
        assert _data_info[-1] is not None
        # Calibration - Get closest calib GPS time and read out
        data_gps_time = int(os.path.split(_data_path)[-1].split("_")[-3])
        nearest_calib_time = calib_gps_times[ifo][np.argmin(np.abs(
            np.array(calib_gps_times[ifo]) - data_gps_time))]
        calib_file = glob.glob(os.path.join(calib_dir, ifo,
                                            f"*{nearest_calib_time}*FinalResults.txt"))[0]
        data = np.loadtxt(calib_file)
        # Assume gaussian error
        freq, mag, mag_1sigma = data[:, 0], data[:, 1], data[:, 5]
        # assert np.max(mag+mag_1sigma) < 1.33
        f_calib_mag = interp1d(freq, mag, bounds_error=False)
        f_calib_sigma = interp1d(freq, np.abs(mag_1sigma - mag), bounds_error=False)
        # "Return"
        _data_info.append((f_calib_mag, f_calib_sigma))
        data_info.append(_data_info)

    # ###### # ###### # ###### #
    test_frequencies = np.logspace(np.log10(tf[0, 1]), np.log10(tf[-1, 1]), n_frequencies)
    def args():
        count = 0
        for test_freq in test_frequencies:
            # Variables to "fill-in" for each data segment
            Y, bkg, model_args, peak_norm, calib_args = [], [], [], [], []
            for hdf_path, ifo, df, (f_calib_mag, f_calib_sigma) in data_info:
                # Find the df entry that contains the tested frequency
                mask = (df['fmin'] <= test_freq) & (df['fmax'] >= test_freq)
                # Skip this entry if test_freq is out of bounds
                if np.sum(np.array(mask, dtype=int)) == 0:
                    continue
                x_knots, y_knots, alpha_skew, loc_skew, sigma_skew, chi_sqr\
                    = df[mask][df_columns].iloc[0]
                # Skip this entry if fit is bad
                if chi_sqr >= max_chi_sqr:
                    continue

                # Find closest matching frequency
                frequencies = hdf_path["frequency"]
                frequency_idx = sensutils.binary_search(frequencies, test_freq)
                # Take care of edge cases
                while frequencies[frequency_idx] < tf[0, 1]:
                    frequency_idx += 1
                while frequencies[frequency_idx] > tf[-1, 1]:
                    frequency_idx -= 1
                # frequency_idx = np.argmin(np.abs(frequencies - test_freq))
                freq_Hz = frequencies[frequency_idx]

                # Get tf factor
                A_star_sqr = f_A_star[ifo](freq_Hz)**2

                # Get calibration mu and sigma
                calib_mu, calib_sigma = f_calib_mag(freq_Hz), f_calib_sigma(freq_Hz)

                # Fill argument lists
                Y.append(hdf_path["logPSD"][frequency_idx])
                bkg.append(models.model_xy_spline(
                    np.concatenate([x_knots, y_knots]), extrapolate=True)(
                        np.log(freq_Hz)))
                model_args.append([alpha_skew, loc_skew, sigma_skew])
                peak_norm.append(rho_local / (np.pi*freq_Hz**3 * A_star_sqr
                                              * (constants.e / constants.h)**2))
                calib_args.append([calib_mu, calib_sigma])

            if len(Y) == 0:
                continue
            yield (count, freq_Hz, do_calib,
                   *(np.array(x) for x in (Y, bkg, model_args, peak_norm, calib_args)))

            count += 1  # Book-keeping for results merging later

    # 2. Create job Pool
    # results = []
    # with Pool(n_processes, maxtasksperchild=10) as pool:
    #     with tqdm(total=n_frequencies, position=0, desc="Looking for DM") as pbar:
    #         for result in pool.imap_unordered(process_segment, args()):
    #             results.append(result)
    #             pbar.update(1)

    # # 3. Merge results
    # discovery_data = np.zeros((2, len(results)))
    # for i, freq_Hz, significance in results:
    #     discovery_data[0, i] = freq_Hz
    #     discovery_data[1, i] = significance

    # tmp
    discovery_data = np.load("discovery.npy")
    # np.save("discovery.npy", discovery_data)

    # 4. Plots & stuff
    ax = plt.subplot(111)
    ax.plot(discovery_data[0, :], discovery_data[1, :])

    # Nice things
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Discovery significance")
    ax.grid(color="grey", alpha=.33, linestyle="--", linewidth=1.5, which="both")
    ax.axhline(5, color="r", linestyle="--")
    mask = np.isnan(discovery_data[0, :])
    ax.set_xlim(min(discovery_data[0, :][~mask]), max(discovery_data[0, :][~mask]))
    ax.minorticks_on()
    plt.show()

    # For each line above a certain significance, plot all data
    # full segments around that frequency bin!
    discovery_threshold = 5  # sigma
    for i in range(discovery_data.shape[1]):
        if discovery_data[1, i] == 0 or np.isnan(discovery_data[1, i]) or\
            np.isinf(discovery_data[1, i]) or discovery_data[1, i] < discovery_threshold:
            continue

        plt.figure()
        ax = plt.subplot(111)
        for hdf_path, ifo, df, (f_calib_mag, f_calib_sigma) in data_info:
            # Find the df entry that contains the tested frequency
            mask = (df['fmin'] <= discovery_data[0, i]) & (df['fmax'] >= discovery_data[0, i])
            fmin, fmax = df[mask][["fmin", "fmax"]].iloc[0]

            # Find closest matching frequency
            frequencies = hdf_path["frequency"]
            fmin_idx = sensutils.binary_search(frequencies, fmin)
            fmax_idx = sensutils.binary_search(frequencies, fmax)

            ax.plot(hdf_path["frequency"][fmin_idx:fmax_idx],
                    hdf_path["logPSD"][fmin_idx:fmax_idx])

        # Nice things
        ax.axvline(discovery_data[0, i], color="r", linestyle="--", linewidth=.5)
        ax.set_title(f"q0: {discovery_data[1, i]:.1f}")
        plt.show()


if __name__ == '__main__':
    main("data", n_processes=9)
