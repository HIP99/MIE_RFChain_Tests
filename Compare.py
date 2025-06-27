import skrf as rf
import os

import matplotlib.pyplot as plt
import numpy as np

from RF_S21 import RF_S21
from RF_Impulse import RF_Impulse
from SURF_Average import SURF_Average

from Setup_S21 import Setup_S21
from Setup_Impulse import Setup_Impulse

setup_s21 = Setup_S21()
setup_impulse = Setup_Impulse()

channel = "003"

fig, ax = plt.subplots()

s21 = RF_S21(channel=channel, setup=setup_s21)
impulse = RF_Impulse(channel=channel, setup=setup_impulse)

s21.plot_s21_filtered(ax=ax, compare=False, f_start=300, f_stop=1200)
# s21.plot_S21(ax=ax, f_start=300, f_stop=1200)
impulse.plot_fft(ax=ax, f_start=300, f_stop=1200)

surf_name = impulse.info['SURF Channel']

surf = SURF_Average(surf = surf_name)

surf.average_over(window=True)

surf.data.tag += " (Windowed)"

surf.plot_fft(ax=ax,f_start=300, f_stop=1200, scale = len(surf)/2)

surf = SURF_Average(surf = surf_name)

surf.average_over(window=False)

surf.plot_fft_smoothed(ax=ax,f_start=300, f_stop=1200, scale = len(surf)/2)


ax.set_title("Frequency Data from VNA, Scope and SURF data")
plt.legend()
plt.show()
