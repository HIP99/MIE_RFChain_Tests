import matplotlib.pyplot as plt
import numpy as np

from RF_S21 import RF_S21
from RF_Impulse import RF_Impulse

from Setup_S21 import Setup_S21
from Setup_Impulse import Setup_Impulse

"""
Simply makes a plot comparing the gain as measured from the VNA s2p and Scope Impulse Response for each channel
"""

setup_s21 = Setup_S21()
setup_impulse = Setup_Impulse()

H_s21_gain = []
H_impulse_gain = []

V_s21_gain = []
V_impulse_gain = []

for i in range(1, 193):
    channel = f"{i:03d}"
    try:
        s21 = RF_S21(channel=channel, setup=setup_s21)
        impulse = RF_Impulse(channel=channel, setup=setup_impulse)
    except Exception as e:
        continue

    if 'H' in str(impulse.info.get('SURF Channel', '')):
        H_s21_gain.append(s21.average_gain(f_start=300, f_stop=1200))
        H_impulse_gain.append(impulse.gain)

    if 'V' in str(impulse.info.get('SURF Channel', '')):
        V_s21_gain.append(s21.average_gain(f_start=300, f_stop=1200))
        V_impulse_gain.append(impulse.gain)

s21_impulse_corr = np.corrcoef(H_s21_gain, H_impulse_gain)[0, 1]
print(f"Correlation for HPol: {abs(s21_impulse_corr):.3f}")

s21_impulse_corr = np.corrcoef(V_s21_gain, V_impulse_gain)[0, 1]
print(f"Correlation for VPol: {abs(s21_impulse_corr):.3f}")

plt.scatter(H_s21_gain, H_impulse_gain, label="HPol")
plt.scatter(V_s21_gain, V_impulse_gain, label="VPol")

plt.xlabel('S21 average gain (dB)')
plt.ylabel('Impulse Gain')
plt.title('S-Parameter vs Impulse response Gain for Polarisations')
plt.grid(True)
plt.legend()
plt.show()