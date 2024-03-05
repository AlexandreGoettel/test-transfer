"""Create config files to run LPSD on condor or locally."""
import os
from string import Template
import argparse
import numpy as np
import h5py
# Project imports
from utils import LPSDVars


def parse_args():
    """Gather dict of cmdl-args parser output."""
    parser = argparse.ArgumentParser()

    LPSD_args = parser.add_argument_group("LPSD args")
    LPSD_args.add_argument("--input-file", type=str, required=True,
                           help="Path to hdf5 input file.")
    LPSD_args.add_argument("--output-file", type=str, default="output.lpsd",
                           help="Path to output file.")
    LPSD_args.add_argument("--run-file", type=str, default="run_lpsd.sh",
                           help="Path to run file.")
    LPSD_args.add_argument("--path-to-lpsd-exec", type=str, default="lpsd-exec",
                           help="Path to LPSD executable.")
    LPSD_args.add_argument("--channel", type=str, default=None,
                           help="Dataset within the hdf5 file. Defaults to" +
                           " X1:GDS-CALIB_STRAIN_CLEAN where X is the input file's first char.")
    LPSD_args.add_argument("--data-length", type=int, default=None,
                           help="Length of the data in s. Defaults to entire file.")
    LPSD_args.add_argument("--fmin", type=float, default=10,
                           help="Minimum frequency in Hz. (default: 10)")
    LPSD_args.add_argument("--fmax", type=float, default=None,
                           help="Maximum frequency in Hz, defaults to fs/2.")
    LPSD_args.add_argument("--fsample", type=float, default=16384,
                           help="Sampling frequency in Hz. (default: 16384)")
    LPSD_args.add_argument("--resolution", type=float, default=1e-6,
                           help="Relative bin width for LPSD. (default: 1e-6)")
    LPSD_args.add_argument("--Jdes", type=int, default=None,
                           help="Set to manually override the number of required frequency bins.")
    LPSD_args.add_argument("--epsilon", type=float, default=10,
                           help="epsilon factor for block approximation in percent. (default: 10)")
    LPSD_args.add_argument("--max-memory-power", type=int, default=26,
                           help="2^n is maximum array length for memory management. (default 26)")

    condor_parser = parser.add_argument_group("Condor args")
    condor_parser.add_argument("--use-condor", action="store_true")
    condor_parser.add_argument("--request-cpus", type=int, default=1)
    condor_parser.add_argument("--request-disk-GB", type=int, default=16)
    condor_parser.add_argument("--request-memory-GB", type=int, default=16)
    condor_parser.add_argument("--accounting-group", type=str,
                               default="ligo.prod.o4.cw.darkmatter.lpsd")
    condor_parser.add_argument("--accounting-group-user", type=str, required=True)
    condor_parser.add_argument("--submit-file", type=str, default="lpsd.submit",
                               help="Path to submit file to write.")
    condor_parser.add_argument("--outdir", type=str, default=".",
                               help="Where to store condor products.")
    return vars(parser.parse_args())


def write_template(args, infile, outfile):
    """Write the args-substituted contents of infile to outfile."""
    with open(os.path.join(os.path.split("__file__")[0],
                           infile), "r") as _f:
        contents = _f.read()
    with open(outfile, "w") as _f:
        _f.write(Template(contents).safe_substitute(args))

def main(args):
    """
    Generates a LPSD run file and, optionally, condor submit file.

    Parameters:
    - args (dict)

    Raises:
    - ValueError: If the data length is insufficient for the provided 'fmin' and 'resolution'.
    """
    # Make sure default cases are treated correctly
    if args["channel"] is None:
        args["channel"] = f"{args['input_file'][0]}1:GDS-CALIB_STRAIN_CLEAN"

    if args["fmax"] is None:
        args["fmax"] = args["fsample"] / 2

    if args["Jdes"] is None:
        lpsd_args = LPSDVars(args["fmin"], args["fmax"], args["resolution"])
        args["Jdes"] = lpsd_args.Jdes

    if args["data_length"] is None:
        with h5py.File(args["input_file"], "r") as _f:
            args["data_length"] = len(_f[args["channel"]]) // args["fsample"]

    # Check if the settings are viable
    if args["data_length"] < 1 / (args["fmin"] * args["resolution"]):
        _fmin = int(np.ceil(1. / (args["data_length"] * args["resolution"])))
        raise ValueError(f"Insufficient data length\nUnder current settings, try fmin >= {_fmin}")

    # Write the run file based on the template
    write_template(args,
                   "templates/lpsd_config.template",
                   args["run_file"]
                   )

    # Write a condor submit file
    if not args["use_condor"]:
        return
    write_template(args,
                   "templates/lpsd_submitfile.template",
                   args["submit_file"]
                   )


if __name__ == '__main__':
    main(parse_args())
