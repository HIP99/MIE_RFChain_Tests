import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import correlate
from scipy.interpolate import UnivariateSpline

from Impulse_Response import Impulse_Response

class Attenuator_Impulse(Impulse_Response):
    """
    This doesn't really need its own class
    The only purpose is to be an impulse response measurement for an attenuator with specfic file naming convention so one can simply input the level of attenuation and it'll find the file
    Assuming naming convention FullChain_{dB}dBAtn
    """
    def __init__(self, dB = 20, *args, **kwargs):

        self.current_dir = Path(__file__).parent

        filepath = self.current_dir / 'data' / 'Scope_Data' / f'FullChain_{dB}dBAtn'

        super().__init__(filepath=filepath, tag = f"Attenuator_{dB}dB")

if __name__ == '__main__':

    atn_ir = Attenuator_Impulse(dB=20)

    # print(atn_ir.group_delay*1e9)
    fig, ax = plt.subplots()
    atn_ir.plot_fft(ax=ax)
    ax.set_title("Scope Impulse Response 20 dB + Cable Gain spectrum")

    plt.show()