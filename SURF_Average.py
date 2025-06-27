import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from SURF_Data import SURF_Data
from SURF_Channel import SURF_Channel
from Waveform import Waveform

class SURF_Average(SURF_Data):
    """
    Each SURF data aquisition should have 100 pickle files of data. Typically an oscillscope would average over based on a trigger.
    This class takes all the 'triggers' aligns the pulses and takes the superposition
    This method is currerently incredibly geared towards pulses rather than any continuous waveform
    """
    def __init__(self, surf:str = "AV1", *args, **kwargs):

        self.surf = surf
        self.get_info()
        self.data = None


    def __len__(self):
        return len(self.data)

    def get_info(self):
        """
        Loads the channel information for self.surf from the 'SURF Channel' column.
        """
        current_dir = Path(__file__).parent
        filepath = current_dir / 'data' / 'Channel_Assignment.csv'

        df = pd.read_csv(filepath)
        # If your SURF Channel column is string, use self.surf; if int, use int(self.surf)
        row = df[df['SURF Channel'] == self.surf]
        if not row.empty:
            self.info = row.iloc[0].to_dict()
        else:
            print(f"Value {self.surf} not found in the 'SURF Channel' column.")

    def average_over(self, length:int = 999, factor : int = None, window = False):
        first_run = SURF_Channel(surf=self.surf, run=0)
        if window:
            first_run.extract_pulse_window()
        self.data = first_run.data
        self.data.tag = f'SURF Channel : {self.info['Channel']}'
        del first_run

        if factor:
            self.data.upsampleFreqDomain(factor=factor)

        for i in range(length):
            try:
                self.cross_correlate(SURF_Channel(surf=self.surf, run=i+1), factor)
            except Exception as e:
                print(f"Error in get_info for {"Surf : " + self.surf+"_Run_"+str(i+1)}: {e}")
                break

        self.data.waveform /= 1000

    def cross_correlate(self, surf_run : SURF_Channel, factor : int = None, window = False):
        if window:
            surf_run.extract_pulse_window()
        compare_data = surf_run.data
        if factor:
            compare_data.upsampleFreqDomain(factor=factor)

        corr = np.correlate(self.data - self.data.mean, compare_data - compare_data.mean, mode='full')
        lags = np.arange(-len(compare_data) + 1, len(self.data))
        max_lag = lags[np.argmax(corr)]

        compare_data.correlation_align(self.data, max_lag)

        self.data.waveform += compare_data

        del surf_run
        del compare_data

    def plot_data(self, ax: plt.Axes=None):
        if ax is None:
            fig, ax = plt.subplots()

        self.data.plot_waveform(ax=ax)
        # ax.set_xlabel('Samples')
        ax.set_ylabel('Raw ADC Counts')

    def plot_data(self, ax: plt.Axes=None):
        if ax is None:
            fig, ax = plt.subplots()

        self.data.plot_waveform(ax=ax)
        # ax.set_xlabel('Samples')
        ax.set_ylabel('Raw ADC Counts')


    def plot_fft(self, ax: plt.Axes=None, f_start=0, f_stop=2000, log = True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        xf = self.data.xf

        mask = (xf >= f_start*1e6) & (xf <= f_stop*1e6)

        self.data.plot_fft(ax=ax, log = log, mask=mask, **kwargs)


    def plot_fft_smoothed(self, ax: plt.Axes=None, f_start=0, f_stop=2000, log = True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        xf = self.data.xf

        mask = (xf >= f_start*1e6) & (xf <= f_stop*1e6)

        self.data.plot_fft_smoothed(ax=ax, log = log, mask=mask, **kwargs)


if __name__ == '__main__':

    surf_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J', 'K', 'L', 'M']
    surf_name = surf_names[0]+'V'

    fig, ax = plt.subplots()

    surf = SURF_Average(surf="IV6")
    surf.average_over()
    surf.plot_fft(ax=ax,f_start=300, f_stop=1200, log=True, scale = len(surf)/2)
    surf.plot_fft_smoothed(ax=ax,f_start=300, f_stop=1200, log=True, scale = len(surf)/2)

    plt.legend()
    plt.show()