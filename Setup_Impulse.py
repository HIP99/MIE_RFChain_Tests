import numpy as np
import matplotlib.pyplot as plt

from Impulse_Response import Impulse_Response
from Attenuator_Impulse import Attenuator_Impulse
from Cable_Impulse import Cable_Impulse

class Setup_Impulse(Impulse_Response):
    """
    This is where we define the auxiliary setup used for impulse response measurements
    This is used in the RF impulse class to adjust readings for how the raw setup performs
    """
    def __init__(self, *args, **kwargs):

        self.atn20 = Attenuator_Impulse(dB=20)
        self.atn30 = Attenuator_Impulse(dB=30)
        self.cable = Cable_Impulse()

        self.tag = 'Setup'

    @property
    def group_delay(self):
        return self.atn20.group_delay + self.atn30.group_delay - self.cable.group_delay
    
    @property
    def gain(self):
        return self.atn20.gain + self.atn30.gain - self.cable.gain
    
    @property
    def fft(self):
        cable_20 = self.atn20.fft
        cable_30 = self.atn30.fft
        cable = self.cable.fft

        overall_fft = cable_20 * cable_30 / cable
        return overall_fft
    
    @property
    def mag_spectrum(self):
        N = len(self.atn20.response)
        overall_fft = self.fft
        overall_mag = np.abs(overall_fft[:N//2 + 1])
        return self.atn20.response.xf, overall_mag

    @property
    def mag_spectrum_db(self):
        _, overall_mag = self.mag_spectrum
        return self.atn20.response.xf, self.lin_to_db(overall_mag)

if __name__ == '__main__':
    setup = Setup_Impulse()
    # print(setup.gain)
    setup.plot_fft(f_start=300, f_stop=1200)
    plt.legend()
    plt.show()
