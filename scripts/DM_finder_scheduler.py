"""Schedule job creation and write submit files for DM hunting."""
import os
import argparse
import numpy as np
# Project imports
import utils
from DM_finder_organiser import create_job_args


BASE_PATH = os.path.split(os.path.abspath(__file__))[0]


def parse_cmdl_args():
    """Parse cmdl args to pass to main."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add arguments
    parser.add_argument("--rundir", type=str, required=True,
                        help="Where to store all run files.")
    parser.add_argument("--skip-args", action="store_true")
    parser.add_argument("--prefix", type=str, required=True,
                        help="Where to save the data (for condor).")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the MC/data input file.")
    parser.add_argument("--data-prefix", type=str, default="result",
                        help="Prefix of data files.")
    parser.add_argument("--json-path", type=str, required=False,
                        help="Path to the post-processing json file.")
    parser.add_argument("--peak-shape-path", type=str, required=True,
                        help="Path to the peak_shape_data.npz file.")
    parser.add_argument("--dname", type=str, required=True, help="Name of the PSD dataset.")
    parser.add_argument("--dname-freq", type=str, required=True,
                        help="Name of the frequency dataset.")
    parser.add_argument("--max-chi", type=int, default=10,
                        help="Maximum chi^2 deviation to skew norm fit in a chunk.")
    parser.add_argument("--fmin", type=float, default=10)
    parser.add_argument("--fmax", type=float, default=5000)
    parser.add_argument("--freqs-per-job", type=int, default=35000)
    parser.add_argument("--isMC", required=False, action="store_true")
    parser.add_argument("--injection-path", type=str, default=None,
                        help="If path is given, add those specified injections using peak shape.")
    parser.add_argument("--injection-peak-shape-path", type=str, default=None,
                        help="If given, use this for injections instead of peak-shape-path.")
    parser.add_argument("--n-processes", type=int, default=4)
    parser.add_argument("--start-iter", type=int, default=0)
    return vars(parser.parse_args())


def write_submit_wrapper(script_path, start_iter=0):
    """Write the condor submit executable - wrapping around the python script."""
    _str = f"""
#!/bin/bash

# The iteration number is passed by HTCondor as the process number
ITERATION=$(({start_iter} + $1))

# Call the Python script
python {script_path} --iteration $ITERATION --n-processes $2 --prefix $3 --peak-shape-path $4
"""
    return _str


def write_submit_file(N_jobs, request_cpus, path_to_wrapper,
                      prefix, out_prefix, peak_shape_path):
    """Write the condor submit file for DM hunting jobs."""
    _str = f"""
Universe = vanilla
Executable = {path_to_wrapper}
Arguments = $(Process) $(NUM_CPUS) {os.path.split(prefix)[-1]} {os.path.split(peak_shape_path)[-1]}
request_cpus = {request_cpus}
transfer_input_files = {prefix}_$(Process).npz {peak_shape_path}
accounting_group = aluk.dev.o4.cw.darkmatter.lpsd
request_disk = 1 GB
request_memory = 4 GB

Output = {out_prefix}_$(Process).out
Error = {out_prefix}_$(Process).err
Log = {out_prefix}_$(Process).log

Queue {N_jobs}
"""
    return _str


def writefile(_str, filename):
    """Write the contents of _str to filename."""
    with open(filename, "w") as _f:
        _f.write(_str)


def main(prefix=None, rundir=None, fmin=10, fmax=5000, freqs_per_job=35000,
         skip_args=True, n_processes=4, start_iter=0, **kwargs):
    """Organise argument creation and job submission."""
    # Analysis variables
    resolution = 1e-6
    fmin_lpsd = 10  # Hz
    fmax_lpsd = 8192  # Hz

    # Process variables
    Jdes = utils.Jdes(fmin_lpsd, fmax_lpsd, resolution)

    # Create job arguments for each freq. block
    fmin_job, fmax_job = [], []
    f = np.logspace(np.log10(fmin_lpsd), np.log10(fmax_lpsd), Jdes)
    N_min = int((np.log(fmin) - np.log(fmin_lpsd)) / np.log(1. + resolution))
    N_max = min(Jdes, int(1 + (np.log(fmax) - np.log(fmin_lpsd)) / np.log(1. + resolution)))

    starting_indices = np.arange(N_min, N_max + freqs_per_job, freqs_per_job)
    if not skip_args:
        print(f"Creating args for {len(starting_indices)} jobs..")
        for idx_min in starting_indices:
            idx_max = min(idx_min + freqs_per_job, N_max)
            if idx_min >= idx_max:
                continue

            fmin_job.append(f[idx_min])
            if idx_max >= Jdes:
                fmax_job.append(fmax)
                break
            fmax_job.append(f[idx_max])
        create_job_args(prefix, fmin_job, fmax_job, **kwargs)

    # Write submit files, coord with executable
    isolated_prefix = os.path.split(prefix)[-1]
    run_prefix = os.path.join(rundir, isolated_prefix)
    path_to_wrapper = f"{run_prefix}_wrapper.sh"
    path_to_executable = os.path.join(BASE_PATH, "DM_finder_executable.py")
    path_to_submitfile = f"{run_prefix}.submit"

    writefile(write_submit_wrapper(path_to_executable, start_iter),
              path_to_wrapper)
    writefile(write_submit_file(len(starting_indices), n_processes, path_to_wrapper,
                                prefix, run_prefix, kwargs["peak_shape_path"]),
              path_to_submitfile)
    print(f"Wrote submit files to {path_to_wrapper} and {path_to_submitfile}.")


if __name__ == '__main__':
    main(**parse_cmdl_args())
