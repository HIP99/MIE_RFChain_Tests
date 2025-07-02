import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Waveform import Waveform

class Scope_Data():
    """
    Method for retrieving data from an oscilloscope. The model of which I forget
    Automatically gets the scopes info for the data taking
    """
    def __init__(self, filepath=None, tag=None, *args, **kwargs):

        keys = [
            "Resolution",
            "RecordLength",
            "XStart",
            "XStop",
            "MultiChannelSource",
            "MultiChannelViewUnit",
            "MultiChannelVerticalScale",
            "XAxisTDRDomain"
        ]

        self.tag = tag

        self.scope_info = {}

        self.get_scope_info(filepath=filepath, keys = keys)

        ##This should always be self.time
        self.time = np.linspace(self.scope_info['XStart'], self.scope_info['XStop'], self.scope_info['RecordLength'])
        # setattr(self, self.scope_info['XAxisTDRDomain'], np.linspace(self.scope_info['XStart'], self.scope_info['XStop'], self.scope_info['RecordLength']))

        data = np.loadtxt(filepath.with_suffix('.Wfm.csv'), delimiter=",")


        data = np.loadtxt(filepath.with_suffix('.Wfm.csv'), delimiter=",", ndmin=2)
        if data.shape[0] < data.shape[1]:
            data = data.T  # Ensure shape is (N_samples, N_channels)

        for i, source in enumerate(self.scope_info['MultiChannelSource']):
            if source:
                setattr(self, source, Waveform(data[:, i], sample_frequency=1/self.scope_info['Resolution'], tag = f"{source} : {self.tag}"))

    def get_scope_info(self, filepath, keys):
        with open(filepath.with_suffix('.csv'), "r") as f:
            for line in f:
                for key in keys:
                    if line.startswith(key + ":"):
                        # Split on colon and strip whitespace/extra colons
                        value = line.split(":", 1)[1].strip().strip(":")
                        if ":" in value:
                            parts = value.split(":")
                            try:
                                parts = [float(x) for x in parts[1:]]
                                self.scope_info[key] = parts
                            except ValueError:
                                parts = [None if x == 'None' else x for x in parts[1:]]
                                self.scope_info[key] = parts
                        else:
                            try:
                                self.scope_info[key] = eval(value)
                            except:
                                self.scope_info[key] = value

    def lin_to_db(self, lin):
        db = 20*np.log10(lin)
        return db

    def plot_data(self, ax: plt.Axes=None):

        channels = [
            getattr(self, source)
            for source in self.scope_info['MultiChannelSource']
            if source
            ]
        
        if ax is None:
            fig, ax = plt.subplots()

        for channel in channels:
            channel.plot_waveform(ax=ax, scale=1e3)

        ax.set_ylabel("Voltage (mV)")
        ax.legend()

    def plot_fft(self, ax: plt.Axes=None, f_start=0, f_stop=2000, log = True, add_ons = False):
        channels = [
            getattr(self, source)
            for source in self.scope_info['MultiChannelSource']
            if source
            ]
        
        if ax is None:
            fig, ax = plt.subplots()

        xf = channels[0].xf

        mask = (xf >= f_start*1e6) & (xf <= f_stop*1e6)

        for channel in channels:
            channel.plot_fft(ax=ax, mask=mask, log = log)


if __name__=="__main__":

    current_dir = Path(__file__).parent

    name = "017"

    ## 061325_endofday_fullchain_referenceimpulse
    
    # filepath = current_dir / 'data' / f'FullChain_{name}'
    filepath = current_dir / 'data' / f'061325_endofday_fullchain_referenceimpulse'

    # filepath='/Users/hpumphrey/Downloads/fullchain_DAQ/'
    # filepath = filepath.with_suffix('.csv')

    data = Scope_Data(filepath=filepath)

    fig, ax = plt.subplots()

    data.C1W1.shorten_waveform(800,850)

    data.plot_fft(ax=ax, log=True)
    # data.plot_data(ax=ax)

    ax.set_title('061325_endofday_fullchain_referenceimpulse fft plot')
    plt.show()