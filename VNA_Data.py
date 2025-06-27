import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import UnivariateSpline

import warnings
warnings.filterwarnings("ignore", category=np.ComplexWarning)

class VNA_Data(rf.Network):
    """
    Extracts data from the s2p file
    Has some methods for handling s2p file although rf.network does a lot of the heavy lifting 
    """
    def __init__(self, filepath, tag = None, *args, **kwargs):

        super().__init__(file = filepath, *args, **kwargs)

        if tag:
            self.tag = tag
        else:
            self.tag = "VNA Measurement"

    def __len__(self):
        return len(self.f)
    
    ######
    # Properties
    ######

    @property
    def s21(self):
        return np.abs(self.s[:, 1, 0])
    
    @property
    def s21_dB(self):
        return (20 * np.log10(self.s21))
    
    @property
    def gd(self):
        return self.group_delay[:, 1, 0]

    ######
    # Setup Methods
    ######

    """When dealing with 192 channels, limiting the data to our bandwidth is a lot quicker"""
    def bandwidth_lin(self, f_start=300, f_stop=1200):
        f_start = f_start * 1e6
        f_stop = f_stop * 1e6

        mask = (self.f >= f_start) & (self.f <= f_stop)
        freq = self.f[mask]
        s21_band = self.s21[mask]
        return freq, s21_band
    
    def bandwidth_db(self, f_start=300, f_stop=1200):
        freq, s21_band = self.bandwidth_lin(f_start=f_start, f_stop=f_stop)
        s21_db_band = (20 * np.log10(s21_band))
        return freq, s21_db_band
    
    def bandwidth_group_delay(self, f_start=300, f_stop=1200):
        f_start = f_start * 1e6
        f_stop = f_stop * 1e6

        gd_band = self.gd * 1e9
        
        mask = (self.f >= f_start) & (self.f <= f_stop)
        freq = self.f[mask]
        gd_band = gd_band[mask]

        return freq, gd_band
    
    def average_gain(self, f_start=300, f_stop=1200):
        _, s21_lin_band = self.bandwidth_lin(f_start, f_stop)

        avg_gain = 20 * np.log10(s21_lin_band.mean())
        return avg_gain
    
    def average_gd(self, f_start=350, f_stop=1150):
        _, gd_s21 = self.bandwidth_group_delay(f_start, f_stop)
        avg_gd = np.mean(gd_s21)
        return abs(avg_gd)
    
    def gain_std(self, f_start, f_stop):
        _, s21_db_lin = self.bandwidth_lin(f_start, f_stop)
        lin_std = s21_db_lin.std()
        return lin_std

    def impulse_response(self):
        """
        This wasn't written properly and doesn't really work
        """
        freq = self.f
        df = freq[1] - freq[0]
        n = len(freq)

        dt = 1 / (n*df)
        t = np.arange(-n//2, n//2) * dt

        # window = np.hanning(n)
        window = np.hamming(n)
        td = np.fft.ifft(self.s[:, 1, 0] * window)
        td = np.fft.fftshift(td)
        return t, td

    def plot_S21(self, f_start=300, f_stop=1200, ax: plt.Axes=None, log = True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        if log:
            freq, bandwidth = self.bandwidth_db(f_start, f_stop)
            ax.set_ylabel('S21 (dB)')
        else:
            freq, bandwidth = self.bandwidth_lin(f_start, f_stop)
            ax.set_ylabel('Magnitude (arb.)')


        ax.plot(freq / 1e6, bandwidth, label = self.tag, **kwargs)

        ax.set_xlabel('Frequency (MHz)')
        ax.set_title('S-parameter response')
        ax.grid(True)

    def plot_S21_Smooth(self, f_start=300, f_stop=1200, ax: plt.Axes=None, label: str = None):
        if ax is None:
            fig, ax = plt.subplots()

        freq, bandwidth_db = self.bandwidth_db(f_start, f_stop)

        spline = UnivariateSpline(freq, bandwidth_db, s=15)
        smoothed_bandwidth_db = spline(freq)

        ax.plot(freq / 1e6, smoothed_bandwidth_db, label=f"{self.tag} (Smoothed)")
        ax.plot(freq / 1e6, bandwidth_db, alpha = 0.5, linestyle = '--', label=f"{self.tag} (original)")

        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('S21 (dB)')
        ax.set_title('Smoothed S21 vs Frequency')
        ax.grid(True)

    def plot_group_delay(self, ax: plt.Axes=None, f_start=300, f_stop=1200, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        freq, gd_s21 = self.bandwidth_group_delay(f_start, f_stop)

        ax.plot(freq/1e6, gd_s21, label = self.tag, **kwargs)
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Group Delay (ns)')
        ax.set_title('Group Delay of VNA measurement')
        ax.grid(True)

    def plot_impulse_response(self, ax: plt.Axes=None, **kwargs):
        """
        Impulse response method isn't very good
        """
        if ax is None:
            fig, ax = plt.subplots()

        t, td = self.impulse_response()

        # first_nonzero_time = t[(t >= 0)][np.argmax(np.abs(td)[t >= 0] > 0.1)]

        mask = slice(None)
        mask = (t >= 0) & (t < 100*1e-9)
        t = t[mask]
        td = td[mask]

        ax.plot(t * 1e9, td, label = self.tag, **kwargs)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Magnitude')
        ax.set_title('Time-Domain Response (S21)')
        ax.grid(True)

if __name__ == '__main__':

    current_dir = Path(__file__).parent
    # name = "GPS_TURF"
    # filepath = current_dir / 'data' / f'{name}.s2p'

    name = "017"
    filepath = current_dir / 'data' / f'fullchain_{name}.s2p'
    filepath = current_dir / 'data' / 'fullchain_30dB+30dB_inlineattenuators.s2p'
    ntw = VNA_Data(filepath=filepath)

    # print(ntw.average_gd(f_start=300, f_stop=1200))

    fig, ax = plt.subplots()
    # ntw.plot_S21(ax = ax, f_start=300, f_stop=1200)
    ntw.plot_group_delay(ax = ax)
    plt.legend()
    plt.show()
