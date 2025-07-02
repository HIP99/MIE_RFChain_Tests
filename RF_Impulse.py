import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from Impulse_Response import Impulse_Response
from Setup_Impulse import Setup_Impulse
from MIE_Channel import MIE_Channel

class RF_Impulse(Impulse_Response, MIE_Channel):
    """
    Impulse response measurement for the AMPA and MIE setup.
    This automatically adjusts the data given a setup of cables and attenuators
    """
    def __init__(self, channel, setup : Setup_Impulse = None, *args, **kwargs):
            self.current_dir = Path(__file__).parent

            filepath = self.current_dir / 'data' / 'Scope_Data' / f'FullChain_{channel}'

            super().__init__(filepath=filepath, tag = f"Scope Channel : {channel}")

            self.info = {"Channel" : channel}
            self.get_info(channel)

            if setup is None:
                self.setup = Setup_Impulse()
            else:
                self.setup = setup

    @property
    def group_delay(self):
        return super().group_delay - self.setup.group_delay
    
    @property
    def gain(self):
        return super().gain
    
    @property
    def fft(self):
        overall_fft = super().fft
        result = overall_fft/self.setup.fft
        return result
    
if __name__ == '__main__':

    ampa = RF_Impulse(channel="017")

    fig, ax = plt.subplots()
    ampa.plot_response(ax=ax)
    plt.show()