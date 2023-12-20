"""Schedule job creation and write submit files for DM hunting."""
import os
import argparse
import numpy as np
# Project imports
import utils


BASE_PATH = os.path.split(os.path.abspath(__file__))[0]


def parse_cmdl_args():
    """Parse cmdl args to pass to main."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add arguments
    parser.add_argument("--rundir", type=str, required=True,
                        help="Where to store all run files.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Where to store the run output.")
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

    parser.add_argument("--fmin", type=float, default=10)
    parser.add_argument("--fmax", type=float, default=5000)
    parser.add_argument("--freqs-per-job", type=int, default=35000)
    parser.add_argument("--isMC", required=False, action="store_true")
    parser.add_argument("--injection-path", type=str, default=None,
                        help="If path is given, add those specified injections using peak shape.")
    parser.add_argument("--injection-peak-shape-path", type=str, default=None,
                        help="If given, use this for injections instead of peak-shape-path.")
    parser.add_argument("--n-processes", type=int, default=4)
    return vars(parser.parse_args())


def write_submit_wrapper(script_path):
    """Write the condor submit executable - wrapping around the python script."""
    _str = f"""#!/bin/bash

# The iteration number is passed by HTCondor as the process number
ITERATION=$(($1 + $2))

# Call the Python script
/home/alexandresebastien.goettel/.conda/envs/scalardarkmatter/bin/python {script_path} --iteration $ITERATION --ana-fmin $3 --ana-fmax $4 --Jdes $5 --data-path $6 --json-path $7 --peak-shape-path $8 --outdir $9 --isMC ${10} --injection-path ${11} --injection-peak-shape-path ${12} --prefix ${13} -n-processes ${14} --n-frequencies ${15}
"""
    return _str


def write_submit_file(N_start, N_end, request_cpus, path_to_wrapper, outdir, prefix,
                      peak_shape_path, n_freqs, ana_fmin, ana_fmax, Jdes, isMC,
                      data_path, json_path, injection_path, injection_peak_shape_path):
    """Write the condor submit file for DM hunting jobs."""
    out_path = os.path.join(outdir, f"{prefix}_$(Process)")
    args = f"{ana_fmin} {ana_fmax} {Jdes} {data_path} {json_path} {peak_shape_path} {outdir}" +\
        f"{isMC} {injection_path} {injection_peak_shape_path} {prefix} {request_cpus} {n_freqs}"
    _str = f"""Universe = vanilla
Executable = {path_to_wrapper}
Arguments = $(Process) {N_start} {args}
request_cpus = {request_cpus}
accounting_group = aluk.dev.o4.cw.darkmatter.lpsd
accounting_group_user = alexandresebastien.goettel
request_disk = 1 GB
request_memory = 4 GB

Output = {out_path}.out
Error = {out_path}.err
Log = {out_path}.log

Queue {N_end + 1 - N_start}
"""
    return _str


def writefile(_str, filename):
    """Write the contents of _str to filename."""
    with open(filename, "w") as _f:
        _f.write(_str)


def main(rundir=None, outdir=None, prefix=None, fmin=10, fmax=5000, freqs_per_job=35000,
         data_path=None, json_path=None, peak_shape_path=None, injection_path=None,
         injection_peak_shape_path=None, n_processes=4, isMC=False, **_):
    """Organise argument creation and job submission."""
    # Analysis variables
    resolution = 1e-6
    fmin_lpsd = 10  # Hz
    fmax_lpsd = 8192  # Hz

    # Process variables
    Jdes = utils.Jdes(fmin_lpsd, fmax_lpsd, resolution)

    freqs = np.logspace(np.log10(fmin_lpsd), np.log10(fmax_lpsd), Jdes)
    iter_start, iter_end = None, None
    for i in range(Jdes//freqs_per_job + 1):
        if iter_start is None and freqs[(i+1)*freqs_per_job] >= fmin:
            iter_start = i
        if iter_end is None and freqs[(i+1)*freqs_per_job] >= fmax:
            iter_end = i
            break
    else:
        raise ValueError

    # Write submit files, coord with executable
    isolated_prefix = os.path.split(prefix)[-1]
    run_prefix = os.path.join(rundir, isolated_prefix)
    path_to_wrapper = f"{run_prefix}_wrapper.sh"
    path_to_executable = os.path.join(BASE_PATH, "DM_finder_executable.py")
    path_to_submitfile = f"{run_prefix}.submit"

    writefile(write_submit_wrapper(path_to_executable), path_to_wrapper)
    writefile(write_submit_file(
        iter_start, iter_end, n_processes, path_to_wrapper, outdir, prefix,
        peak_shape_path, freqs_per_job, fmin_lpsd, fmax_lpsd, Jdes, isMC,
        data_path, json_path, injection_path, injection_peak_shape_path),
              path_to_submitfile)
    print(f"Wrote submit files to {path_to_wrapper} and {path_to_submitfile}")


if __name__ == '__main__':
    main(**parse_cmdl_args())
