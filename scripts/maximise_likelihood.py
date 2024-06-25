"""Schedule job creation and write submit files for DM hunting."""
import os
import argparse
import glob
import numpy as np
# Project imports
from LPSDIO import LPSDOutput


BASE_PATH = os.path.split(os.path.abspath(__file__))[0]


def parse_cmdl_args():
    """Parse cmdl args to pass to main."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add arguments
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the MC/data input file.")
    parser.add_argument("--data-prefix", type=str, default=None,
                        help="Prefix of data files.")
    parser.add_argument("--rundir", type=str, required=True,
                        help="Where to store all run files.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Where to store the run output.")
    parser.add_argument("--prefix", type=str, required=True,
                        help="Run file prefix in outdir (for condor).")
    parser.add_argument("--bkg-info-path", type=str, required=False,
                        help="Output data from fit_background.py.")
    parser.add_argument("--peak-shape-path", type=str, required=True,
                        help="Path to the peak_shape_data.npz file.")
    parser.add_argument("--python-executable", type=str,
                        default="/home/alexandresebastien.goettel/.conda/envs/scalardarkmatter/bin/python",
                        help="Path to python executable from desired env.")
    parser.add_argument("--tf-path", type=str, required=True,
                        help="Path to directory holding transfer functions.")

    parser.add_argument("--ana-fmin", type=float, default=10)
    parser.add_argument("--ana-fmax", type=float, default=5000)
    parser.add_argument("--freqs-per-job", type=int, default=35000)
    parser.add_argument("--n-processes", type=int, default=4)
    parser.add_argument("--accounting-group-user", type=str, required=True)
    parser.add_argument("--accounting-group", type=str, default="ligo.devd.o4.cw.darkmatter.lpsd")
    # aluk.dev.o4.cw.darkmatter.lpsd
    # parser.add_argument("--isMC", required=False, action="store_true")
    # parser.add_argument("--injection-path", type=str, default=None,
    #                     help="If path is given, add those specified injections using peak shape.")
    # parser.add_argument("--injection-peak-shape-path", type=str, default=None,
    #                     help="If given, use this for injections instead of peak-shape-path.")
    return vars(parser.parse_args())


def write_submit_wrapper(python_executable, script_path):
    """Write the condor submit executable - wrapping around the python script."""
    _str = f"""#!/bin/bash

# The iteration number is passed by HTCondor as the process number
ITERATION=$(($1 + $2))

# Call the Python script
{python_executable} {script_path} --iteration $ITERATION"""
    names = ["ana-fmin", "ana-fmax", "data-path", "data-prefix", "bkg-info-path",
             "peak-shape-file", "output-path", "n-processes", "n-frequencies", "tf-path"]
    for i, name in enumerate(names):
        _str += f" --{name} ${{{i+3}}}"
    return _str + "\n"


def write_submit_file(N_start, N_end, request_cpus, path_to_wrapper, outdir, prefix,
                      peak_shape_path, n_freqs, ana_fmin, ana_fmax, json_path, tf_path,
                      data_path, data_prefix, accounting_group, accounting_group_user):
    """Write the condor submit file for DM hunting jobs."""
    out_path = os.path.join(outdir, f"{prefix}_$(Process)")
    args = " ".join([f"{x}" for x in [ana_fmin, ana_fmax, data_path, data_prefix, json_path,
                                      peak_shape_path, os.path.join(outdir, prefix), request_cpus,
                                      n_freqs, tf_path]])
    _str = f"""Universe = vanilla
Executable = {path_to_wrapper}
Arguments = $(Process) {N_start} {args}
request_cpus = {request_cpus}
accounting_group = {accounting_group}
accounting_group_user = {accounting_group_user}
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


def main(rundir=None, outdir=None, prefix=None, ana_fmin=10, ana_fmax=5000, freqs_per_job=35000,
         data_path=None, bkg_info_path=None, peak_shape_path=None, data_prefix="", n_processes=4,
         accounting_group="", accounting_group_user="", python_executable=None,
         tf_path=None, **_):
    """Organise argument creation and job submission."""
    path = glob.glob(os.path.join(data_path, f"{data_prefix}*"))[0] if data_prefix is not None else data_path
    data = LPSDOutput(path)

    iter_start, iter_end = None, None
    for i in range(len(data)//freqs_per_job + 1):
        if iter_start is None and data.freq[(i+1)*freqs_per_job] >= ana_fmin:
            iter_start = i
        if iter_end is None and data.freq[(i+1)*freqs_per_job] >= ana_fmax:
            iter_end = i
            break
    else:
        raise ValueError

    # Write submit files, coord with executable
    isolated_prefix = os.path.split(prefix)[-1]
    run_prefix = os.path.join(rundir, isolated_prefix)
    path_to_wrapper = f"{run_prefix}_wrapper.sh"
    path_to_executable = os.path.join(BASE_PATH, "max_lkl_executable.py")
    path_to_submitfile = f"{run_prefix}.submit"

    writefile(write_submit_wrapper(python_executable, path_to_executable), path_to_wrapper)
    writefile(write_submit_file(
        iter_start, iter_end, n_processes, path_to_wrapper, outdir, prefix,
        peak_shape_path, freqs_per_job, ana_fmin, ana_fmax, bkg_info_path, tf_path,
        data_path, data_prefix, accounting_group, accounting_group_user), path_to_submitfile)

    print(f"Wrote submit files to {path_to_wrapper} and {path_to_submitfile}")


if __name__ == '__main__':
    main(**parse_cmdl_args())
