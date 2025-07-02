import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from Impulse_Response import Impulse_Response
from Setup_Impulse import Setup_Impulse

class RF_Impulse(Impulse_Response):
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

    def __str__(self):
        return (
            f"Channel: {self.info["Channel"]}\n"
            f"AMPA: {self.info["AMPA"]}\n"
            f"Antenna: {self.info["Antenna"]}\n"
            f"Phi Sector: {self.info["Phi Sector"]}\n"
            f"SURF Channel: {self.info["SURF Channel"]}"
        )

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

    def get_info(self, channel):
        """
        This just loads in the channel infomation based on the channel number
        """
        filepath = self.current_dir / 'data' / 'Channel_Assignment.csv'


        df = pd.read_csv(filepath)

        row = df[df['Channel'] == int(channel)]
        if not row.empty:
            self.info = row.iloc[0].to_dict()
        else:
            print(f"Value {channel} not found in the 'SURF Channel' column.")
    
if __name__ == '__main__':

    ampa = RF_Impulse(channel="017")

    fig, ax = plt.subplots()
    ampa.plot_response(ax=ax)
    plt.show()