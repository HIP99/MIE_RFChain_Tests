import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import correlate
from scipy.interpolate import UnivariateSpline

from Impulse_Response import Impulse_Response
from Attenuator_Impulse import Attenuator_Impulse

class Cable_Impulse(Impulse_Response):
    """
    The only purpose is to be an impulse response measurement for the cable setup.
    """
    def __init__(self, *args, **kwargs):

        self.current_dir = Path(__file__).parent

        filepath = self.current_dir / 'data' / 'Scope_Data' / f'FullChain_Cables'

        super().__init__(filepath=filepath, tag='Cable')


if __name__ == '__main__':

    cbl_ir = Cable_Impulse()


    print(cbl_ir.group_delay)
    fig, ax = plt.subplots()
    cbl_ir.plot_fft(ax=ax)
    ax.set_title("Scope Impulse Response Cable Gain spectrum")
    # plt.show()

    # cbl_ir.plot_data()

    plt.show()