import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Data import SURF_Data
from Waveform import Waveform
from Pulse import Pulse


class SURF_Channel(SURF_Data):
    """
    Surf data is extracted for everything single surf channel (224 total)
    SURF channel only needs the name of the surf channel and it will extract that surfs data from the whole
    """
    def __init__(self, surf:str = "AV1", run=0, *args, **kwargs):

        self.surf = surf
        self.run = run
        self.data = None

        self.current_dir = Path(__file__).parent

        filepath = self.current_dir / 'data' / 'SURF_Data' / f'SURF{surf}' / f'SURF{surf}_{run}.pkl'

        super().__init__(filepath = filepath, *args, **kwargs)

        self.format_data()


    def __len__(self):
        return len(self.data)

    @property
    def surf_name(self):
        return self.surf[:-1]
    
    @property
    def channel_num(self):
        return int(self.surf[-1])

    def surf_index(self):
        surf_index = self.surf_mapping.index(self.surf_name)
        channel_index = self.channel_mapping[self.channel_num-1]

        return surf_index, channel_index
    
    def format_data(self):
        all_data = super().format_data()

        surf_index, channel_index = self.surf_index()
        self.data = Pulse(waveform=all_data[surf_index][channel_index], sample_frequency=3e9, tag = f'{self.surf}_{self.run}')

    def plot_data(self, ax: plt.Axes=None, *args, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()


        self.data.plot_waveform(ax=ax,**kwargs)
        # ax.plot(self.data, label = f'{self.surf}_{self.run}', **kwargs)
        # ax.set_xlabel('Sample number')
        ax.set_ylabel('Raw ADC counts')


    def plot_fft(self, ax: plt.Axes=None, f_start=0, f_stop=2000, log = True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        xf = self.data.xf

        mask = (xf >= f_start*1e6) & (xf <= f_stop*1e6)

        self.data.plot_fft(ax=ax, log = log, mask=mask, **kwargs)

    def extract_pulse_window(self, pre=20, post=120):
        self.data.pulse_window(pre=20, post=120)

if __name__ == '__main__':
    surf = "AV8"
    run0 = SURF_Channel(surf=surf, run=0)
    run1 = SURF_Channel(surf=surf, run=1)
    run2 = SURF_Channel(surf=surf, run=2)

    fig, ax = plt.subplots()

    run0.plot_fft(ax=ax, log=True)

    plt.legend()
    plt.show()
