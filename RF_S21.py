import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from VNA_Data import VNA_Data
from Setup_S21 import Setup_S21

import warnings
warnings.filterwarnings("ignore", category=np.ComplexWarning)

class RF_S21(VNA_Data):
    """
    VNA S21 measurement for the AMPA and MIE setup.
    This automatically adjusts the data given a setup of cables and attenuators
    The VNA data has a lot of ripples and there's a lot of subsidiary methods relating to the ripples
    """
    def __init__(self, channel, setup : Setup_S21 = None, *args, **kwargs):

        self.current_dir = Path(__file__).parent

        filepath = self.current_dir / 'data' / 'VNA_Data' / f'fullchain_{channel}.s2p'

        super().__init__(filepath = filepath, *args, **kwargs)

        self.info = {"Channel" : channel}
        self.get_info(channel)

        self.tag = f"VNA Channel : {channel}"

        if setup:
            self.setup = setup
        else:
            self.setup = Setup_S21()

    ######
    # Magic methods
    ######

    def __str__(self):
        return (
            f"Channel: {self.info['Channel']}\n"
            f"AMPA: {self.info['AMPA']}\n"
            f"Antenna: {self.info['Antenna']}\n"
            f"Phi Sector: {self.info['Phi Sector']}\n"
            f"SURF Channel: {self.info['SURF Channel']}"
        )
    
    def __len__(self):
        return len(self.f)
    
    ######
    # Properties
    ######

    @property
    def s21(self):
        s21 = super().s21
        return s21 / self.setup.s21
    
    @property
    def s21_dB(self):
        s21_dB = super().s21_dB
        return s21_dB - self.setup.s21_dB
    
    @property
    def gd(self):
        gd = super().gd
        return gd - self.setup.gd
    
    ######
    # Key Methods
    ######

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
    
    ######
    # Misceleneous
    ######

    ##This only seems to work in our bandwidth
    def stupid_filter(self, freq, bandwidth_db, ripple_period):
        n = len(bandwidth_db)

        # Detrend (remove baseline)
        p = np.polyfit(freq, bandwidth_db, deg=2)
        baseline = np.polyval(p, freq)
        s21_detrended = bandwidth_db - baseline

        # FFT
        fft_vals = np.fft.fft(s21_detrended)
        fft_freqs = np.fft.fftfreq(n, d=(freq[1] - freq[0]))

        ripple_freq = 1 / (ripple_period * 1e6)  # Hz^-1
        idx = np.argmin(np.abs(np.abs(fft_freqs) - ripple_freq))

        # Notch out the ripple and its negative frequency
        width = 2  # Notch width in bins, adjust as needed
        fft_vals[idx-width:idx+width+1] = 0
        fft_vals[-idx-width:-idx+width+1] = 0

        # Inverse FFT to get filtered S21
        s21_filtered = np.fft.ifft(fft_vals).real + baseline

        return s21_filtered
    
    ######
    # Ripple stuff
    ######

    def ripple_peaks_troughs(self, f_start=800, f_stop=1100, settings: object = {"prominence":0.001, "distance": 15}):

        freq, s21_db_band = self.bandwidth_db(f_start, f_stop)

        spline = UnivariateSpline(freq, s21_db_band, s=1e2)
        baseline = spline(freq)

        s21_db_detrended = s21_db_band - baseline

        peaks, _ = find_peaks(s21_db_detrended, **settings)
        troughs, _ = find_peaks(-s21_db_detrended, **settings)

        paired_peaks = []
        paired_troughs = []

        troughs_idx = 0
        for peak in peaks:
            # Find the first trough after this peak
            while troughs_idx < len(troughs) and troughs[troughs_idx] < peak:
                troughs_idx += 1
            if troughs_idx < len(troughs):
                paired_peaks.append(peak)
                paired_troughs.append(troughs[troughs_idx])
                troughs_idx += 1

        return freq, s21_db_band, s21_db_detrended, paired_peaks, paired_troughs
    
    def ripple_amp_period_phase(self, f_start=800, f_stop=1100, settings: object = {"prominence":0.001, "distance": 15}):
        freq, s21_db_band, s21_db_detrended, peaks, troughs = self.ripple_peaks_troughs(f_start, f_stop, settings)

        peak_freqs = freq[peaks]
        trough_freqs = freq[troughs]

        periods = []
        amps = []

        for i in range(len(peaks)-1):
            peak_period = peak_freqs[i+1]-peak_freqs[i]
            trough_period = trough_freqs[i+1]-trough_freqs[i]
            periods.append((peak_period + trough_period) / (2*1e6))  # Convert to MHz

        amps = s21_db_detrended[peaks] - s21_db_detrended[troughs]

        return peak_freqs[0]/1e6, np.array(periods), np.array(amps)


    ######
    # Analysis stuff
    ######


    ######
    # Plots
    ######

    def plot_s21_filtered(self, f_start=300, f_stop=1200, filter_periods : list = [17.3, 80, 90], ax: plt.Axes=None, compare=True):
        if ax is None:
            fig, ax = plt.subplots()

        freq, bandwidth_db = self.bandwidth_db(f_start, f_stop)

        s21_filtered = self.stupid_filter(freq, bandwidth_db, 17.3)

        s21_filtered = self.stupid_filter(freq, s21_filtered, 80)

        s21_filtered = self.stupid_filter(freq, s21_filtered, 90)

        if compare:
            ax.plot(freq / 1e6, bandwidth_db, label=f"{self.tag} (original)", alpha=0.5)
        ax.plot(freq / 1e6, s21_filtered, label=f"{self.tag} (filtered)", linewidth=2)
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('S21 (dB)')
        ax.set_title('S21 with Ripple Filtered Out')
        ax.grid(True)

    ######
    # Ripple Related plots
    ######

    def plot_ripple_fft(self, f_start=300, f_stop=1200, ax: plt.Axes=None, **kwargs):
        """
        Treating the s-parameter as a waveform to look at the ripples
        """
        if ax is None:
            fig, ax = plt.subplots()

        freq, s21_db_band = self.bandwidth_db(f_start, f_stop)

        n = len(s21_db_band)
        s21_db_band -= np.mean(s21_db_band)  # Remove DC offset
        fft_vals = np.fft.fft(s21_db_band)
        fft_freqs = np.fft.fftfreq(n, d=(freq[1]-freq[0]))
        
        freq_axis = 1/fft_freqs[1:n//2]/1e6

        mask = freq_axis < 150

        ax.plot(freq_axis[mask], np.abs(fft_vals[1:n//2])[mask], label = self.tag)
        ax.set_xlabel('Ripple Period in frequency domain (MHz)')
        ax.set_ylabel('FFT Magnitude')
        ax.set_title('FFT of S21(dB)')
        ax.grid(True)

    def plot_gd_ripple_fft(self, f_start=300, f_stop=1200, ax: plt.Axes=None, **kwargs):
        """
        Treating the s-parameter as a waveform to look at the ripples
        """
        if ax is None:
            fig, ax = plt.subplots()

        freq, group_delay = self.bandwidth_group_delay(f_start, f_stop)

        n = len(group_delay)
        group_delay -= np.mean(group_delay)  # Remove DC offset
        fft_vals = np.fft.fft(group_delay)
        fft_freqs = np.fft.fftfreq(n, d=(freq[1]-freq[0]))
        
        freq_axis = 1/fft_freqs[1:n//2]/1e6

        mask = freq_axis < 150

        ax.plot(freq_axis[mask], np.abs(fft_vals[1:n//2])[mask], label = self.tag)
        ax.set_xlabel('Ripple Period in frequency domain (MHz)')
        ax.set_ylabel('FFT Magnitude')
        ax.set_title('FFT of Group Delay (arb.)')
        ax.grid(True)

if __name__ == '__main__':
    rf_chain = RF_S21("017")

    ##This finds the ripples with longer wavelength
    long_settings = {"prominence" : 0.0005, "distance" : 55, "width" : 1}
    ##This finds the ripples with shorter wavelength
    short_settings = {"prominence" : 0.001, "distance" : 15}

    fig, ax = plt.subplots()

    rf_chain.plot_s21_filtered(ax = ax)
    plt.legend()
    plt.show()
