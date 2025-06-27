import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft, ifft, rfft, irfft
from scipy.interpolate import UnivariateSpline

class Waveform():
    """
    Generic waveform class
    Handles properties of a waveform
    Calculates waveform properties
    Plots waveform data
    """
    def __init__(self, waveform: np.ndarray, time: list  = None, sample_frequency : int = 3e9, tag: str = None, *args, **kwargs):

        self._waveform = waveform
        self._N = len(waveform)

        if time is not None:
            self._time = time
        else:
            self.time = np.linspace(0, self.N/sample_frequency, self.N)
        
        self.tag = tag

    #####
    ## Magic Methods
    #####

    def __len__(self):
        return len(self.waveform)
    
    def __getitem__(self, index):
        return self.waveform[index]
    
    def __setitem__(self, index, value):
        self.waveform[index] = value

    def __iter__(self):
        return iter(self.waveform)
    
    def __add__(self, other):
        return self.waveform + other

    def __sub__(self, other):
        return self.waveform - other

    def __mul__(self, other):
        return self.waveform * other

    def __truediv__(self, other):
        return self.waveform / other
    
    def __array__(self):
        return self.waveform

    #####
    ## Properties
    #####

    @property
    def waveform(self):
        return self._waveform

    @waveform.setter
    def waveform(self, arr):
        if not isinstance(arr, np.ndarray):
            raise ValueError("Waveform must be of type ndarray")
        self._waveform = arr
        self._N = len(self._waveform)

    @property
    def time(self):
        return self._time
    
    @time.setter
    def time(self, time : list):
        if len(time) < 3:
            raise ValueError("Waveform must an array of some kind")
        self._time = time

    @property
    def sample_rate(self):
        return 1/self.dt

    @property
    def dt(self):
        return self._time[1]-self._time[0]

    @property
    def N(self):
        return self._N
    
    @property
    def xf(self):
        return np.linspace(0.0, 1 / (2 * self.dt), self.N // 2 + 1)
    
    @property
    def fft(self):
        return fft(self.waveform)

    @property
    def ifft(self):
        return ifft(self.waveform)
    
    @property
    def rfft(self):
        return rfft(self.waveform)
    
    @property
    def irfft(self):
        return irfft(self.waveform)

    @property
    def mag_spectrum(self):
        # return np.abs(self.fft[:self.N//2+1]) * 2 / self.N
        return np.abs(self.rfft) * 2 / self.N
    
    @property
    def mean(self):
        return np.mean(self.waveform)
    
    @property
    def p2p(self):
        return abs(max(self.waveform) - min(self.waveform))/2
    
    @property
    def rms(self):
        square_sum = sum(x ** 2 for x in self.waveform)
        mean_square = square_sum / self.N
        return np.sqrt(mean_square)
    
    #####
    ## methods
    #####

    def upsampleFreqDomain(self, factor):
        """
        Shamelessly stolen from Eric
        """
        int_factor = int(factor)-1

        ampl = 2.0*self.rfft
        freq = self.xf
        df = freq[1]-freq[0]

        length = self.N // 2 + 1
        
        ampl = np.pad(ampl, (0, (length-1)*int_factor), 'constant', constant_values=(0))
        freq = np.append(freq, np.arange(freq[-1]+df, length*df,df))

        new_nyquist_ghz = freq[-1]
        self.waveform = factor/2.0 * irfft(ampl)
        new_len = len(self.waveform)
        dt = 1/(2.*new_nyquist_ghz)
        self.time = np.arange(0, new_len) * dt

    def correlation_align(self, other_waveform: "Waveform", max_lag: int):
        """
        Aligns this waveform to anothers based on the maximum correlation on lag
        """
        target_len = len(other_waveform)
        if max_lag > 0:
            # self lags: pad at start
            aligned = np.pad(self.waveform, (max_lag, 0), mode='constant')[:target_len]
        elif max_lag < 0:
            # self leads: trim start
            aligned = self.waveform[-max_lag:]
            if len(aligned) < target_len:
                aligned = np.pad(aligned, (0, target_len - len(aligned)), mode='constant')
            else:
                aligned = aligned[:target_len]
        else:
            aligned = self.waveform[:target_len]
        self.waveform = aligned
        
    #####
    ## Clock formatting stuff
    #####

    def shorten_waveform(self, start, end, relative_time=True):
        """
        Shortens the waveform instance, with the default option to reset time to 0
        """
        self.waveform = self.waveform[start:end]
        if relative_time:
            dt = self.dt  # time step remains the same
            self.time = np.arange(0, len(self.waveform)) * dt
        else:
            self.time = self.time[start:end]

    #####
    ## Plots
    #####

    def plot_waveform(self, ax: plt.Axes=None, scale = 1.0, mask = slice(None), **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        ##Convert to nano seconds
        ax.plot(self.time[mask]*1e9, self.waveform[mask]*scale, label = self.tag, **kwargs)

        ax.set_xlabel('time (ns)')
        
    def plot_fft(self, ax: plt.Axes=None, scale = 1.0, mask = slice(None), log=False, gain=0.0, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        mag = self.mag_spectrum[mask] * scale

        if log:
            mag = 20 * np.log10(mag + 1e-12) + gain

        ax.plot(self.xf[mask]/1e6, mag, label=self.tag, **kwargs)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Magnitude" + (" (dB)" if log else ""))
        ax.set_title("FFT Magnitude Spectrum")
        ax.grid(True)

    def plot_fft_smoothed(self, ax: plt.Axes=None, scale=1.0, mask=slice(None), log=False, window_size=11, gain=0.0, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        mag = self.mag_spectrum[mask] * scale

        window = np.ones(window_size) / window_size
        mag_smoothed = np.convolve(mag, window, mode='same')

        if log:
            mag_smoothed = 20 * np.log10(mag_smoothed + 1e-12) + gain

        ax.plot(self.xf[mask] / 1e6, mag_smoothed, label=f"{self.tag} (smoothed)", **kwargs)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Magnitude" + (" (dB)" if log else ""))
        ax.set_title("Smoothed FFT Magnitude Spectrum")
        ax.grid(True)
    