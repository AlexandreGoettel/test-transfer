"""Calculate LPSD's sensitivity by injecting different signals in realistic noise."""
# Project imports
from sensparser import parse_inputs
from fakedata import DataManager
from output import write_lpsd_sh
# from peakfinder import PeakFinder


def main(**kwargs):
    """Define PSD, injection parameters, and work depending on user input."""
    # Define noise PSD
    manager = DataManager(**kwargs)

    # Define injections (if any)
    if kwargs["analysis"] == "generate":
        #   - Generate time series from noise+injections
        #   - Write to file
        manager.generate_noise()
        manager.plot_data()
        manager.add_injections()  # TODO
        manager.save_data()
        if kwargs["gen_lpsd_sh"]:
            write_lpsd_sh(**kwargs)
    elif kwargs["analysis"] == "analyse":
        pass
        #   - Read LPSD output
        #   - Plot sensitivity over frequency and amplitude


if __name__ == '__main__':
    main(**vars(parse_inputs()))
