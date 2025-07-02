import matplotlib.pyplot as plt
import numpy as np

from RF_S21 import RF_S21
from RF_Impulse import RF_Impulse

from Setup_S21 import Setup_S21
from Setup_Impulse import Setup_Impulse

"""
Simply makes a plot comparing the group delay as measured from the VNA s2p and Scope Impulse Response for each channel
"""

setup_s21 = Setup_S21()
setup_impulse = Setup_Impulse()

H_s21_group = []
H_impulse_group = []

V_s21_group = []
V_impulse_group = []

for i in range(1, 193):
    channel = f"{i:03d}"
    try:
        s21 = RF_S21(channel=channel, setup=setup_s21)
        impulse = RF_Impulse(channel=channel, setup=setup_impulse)
    except Exception as e:
        continue

    if 'H' in str(impulse.info.get('SURF Channel', '')):
        H_s21_group.append(s21.average_gd(f_start=350, f_stop=1150))
        H_impulse_group.append(impulse.group_delay*1e9)

    if 'V' in str(impulse.info.get('SURF Channel', '')):
        V_s21_group.append(s21.average_gd(f_start=350, f_stop=1150))
        V_impulse_group.append(impulse.group_delay*1e9)

s21_impulse_corr = np.corrcoef(H_s21_group, H_impulse_group)[0, 1]
print(f"Correlation for HPol: {abs(s21_impulse_corr):.3f}")

s21_impulse_corr = np.corrcoef(V_s21_group, V_impulse_group)[0, 1]
print(f"Correlation for VPol: {abs(s21_impulse_corr):.3f}")

plt.scatter(H_s21_group, H_impulse_group, label="HPol")
plt.scatter(V_s21_group, V_impulse_group, label="VPol")

plt.xlabel('S21 Group Delay (average)')
plt.ylabel('Impulse Group Delay')
plt.title('S-Parameter vs Impulse response Group Delay for Polarisations')
plt.grid(True)
plt.legend()
plt.show()