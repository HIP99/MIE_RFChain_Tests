import numpy as np
import matplotlib.pyplot as plt

from Waveform import Waveform

class Pulse(Waveform):
    """
    This arguably could use a lot more methods relating to Pulses
    Inherits from Waveform but specifies that the waveform includes a pulse measurement
    """
    def __init__(self, waveform: np.ndarray, time: list  = None, sample_frequency : int = 3e9, tag: str = None, role = "pulse", *args, **kwargs):

        super().__init__(time=time, waveform=waveform, sample_frequency=sample_frequency, tag=tag, *args, **kwargs)

        self.role = role
        self.waveform -= self.offset

        if self.tag is not None:
            self.tag = f"{self.tag}_{self.role}"


    @property
    def offset(self):
        ##This is really specific but finding this range is somewhat taxing, espicially since it's the same for all pulses
        if self.role == 'pulse':
            mask = self.time >= 40*1e-9
        else:
            mask = self.time <= 40*1e-9
        return np.mean(self.waveform[mask])
    

    def find_pulse_peak(self):
        idx = np.argmax(self.waveform)
        return self.time[idx], self.waveform[idx]
    

    def pulse_window(self, pre=20, post=120):
        """
        This simply windows the pulse around the window. 
        It has a very rudamentary method of finding the pulse but does a decent enough job
        The inputs heavily depend on the data extracted
        """
        valid_range = (pre, 1024-post)
        data_abs = np.abs(self.waveform).copy()
        tried = set()
        while True:
            max_idx = np.argmax(data_abs)
            if max_idx in tried:
                print("No pulse found in valid range.")
                return None
            if valid_range[0] <= max_idx <= valid_range[1]:
                start = max(0, max_idx - pre)
                end = min(len(self), max_idx + post)
                self.shorten_waveform(start,end)
                return self.waveform
            # Mask out this index and try again
            data_abs[max_idx] = 0
            tried.add(max_idx)
