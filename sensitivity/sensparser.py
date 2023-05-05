"""Implement parser for sensitivity.py."""
import argparse


def parse_inputs():
    """Parse the command line inputs and return them in a dict, if used."""
    parser = argparse.ArgumentParser(
        prog="sensitivity",
        description="Sensitivity calculator for the o3 LIGO LPSD project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    reqParser = parser.add_argument_group("Required args")
    reqParser.add_argument("--analysis", type=str, required=True,
                           choices=["generate", "analyse"],
                           help="Run the code to generate injections or to read LPSD output?")

    injParser = parser.add_argument_group("Injection args")
    injParser.add_argument("--wavetype", type=str, required=False,
                           choices=["sinelike", "DMlike"], default="DMlike",
                           help="Inject sine waves or DM-type waves?")
    injParser.add_argument("--injection-type", type=str, required=False,
                           choices=["None", "from-file", "given-frequencies", "logspace"],
                           help="Where to place injections. If 'from-file' use --injection-file," +
                           "if 'given-frequencies' use --injection-frequencies.")
    injParser.add_argument("--injection-file", type=str, required=False,
                           help="Filename containing frequencies at which to generate injections.")
    injParser.add_argument("--injection-frequencies", type=float, required=False,
                           nargs="*", default=[],
                           help="List of frequencies if --injection-type is 'given-frequencies'")

    noiseParser = parser.add_argument_group("Noise generation args")
    noiseParser.add_argument("--noise-source", type=str, required=False,
                             default="spline", choices=["spline", "data"],
                             help="From spline-based PSD or from data?")
    noiseParser.add_argument("--psd-file", type=str, required=False,
                             help="If noise-source=='file', read from this one.")
    noiseParser.add_argument("--length", type=int, required=False,
                             default=60, help="Length of the timeseries in s.")
    noiseParser.add_argument("--sampling-frequency", type=float, required=False,
                             default=16384., help="Time domain sampling frequency in Hz")
    noiseParser.add_argument("--output-file", type=str,
                             required=False, default="",
                             help="Where to store generated time series, must be .h5 or .hdf5")

    LPSDParser = parser.add_argument_group("LPSD .sh args")
    LPSDParser.add_argument("--gen-lpsd-sh", action="store_true",
                            help="Whether to generate an .sh file for LPSD.")
    LPSDParser.add_argument("--fmin", type=float, required=False,
                            default=10.)
    LPSDParser.add_argument("--fmax", type=float, required=False,
                            default=8192.)
    LPSDParser.add_argument("--resolution", type=float, required=False,
                            default=1e-6)
    LPSDParser.add_argument("--path-to-lpsd-exec", type=str, required=False,
                            default="")
    LPSDParser.add_argument("--sh_filename", type=str, required=False,
                            default="")
    return parser.parse_args()
