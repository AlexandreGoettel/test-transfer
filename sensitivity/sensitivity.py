"""Calculate LPSD's sensitivity by injecting different signals in realistic noise."""
# Project imports
from sensparser import parse_inputs
from fakedata import DataManager
# from analyse import Datalyzer
from output import write_lpsd_sh


def main(**kwargs):
    """Define PSD, injection parameters, and work depending on user input."""
    if kwargs["analysis"] == "generate":
        # Generate time series from noise+injections
        manager = DataManager(**kwargs)
        manager.generate_noise()
        manager.add_injections()
        manager.plot_data()
        if kwargs["gen_lpsd_sh"]:
            write_lpsd_sh(**kwargs)
        print("Done!")

    # elif kwargs["analysis"] == "analyse":
        # Read LPSD output
        # manager = Datalyzer(**kwargs)
        # manager.get_lpsd_metadata()
        # manager.get_lpsd_output()
        # manager.plot_PSD()
        # Call peak finder
        # manager.find_peaks()
        # TODO Plot sensitivity over frequency and amplitude


if __name__ == '__main__':
    main(**vars(parse_inputs()))
