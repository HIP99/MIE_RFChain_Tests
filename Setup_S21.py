import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import UnivariateSpline
from VNA_Data import VNA_Data
from functools import cached_property


class Setup_S21(VNA_Data):
    """
    This is where we define the auxiliary setup used for VNA S parameter measurements
    This is used in the RF S21 class to adjust readings for how the raw setup performs
    """
    def __init__(self, *args, **kwargs):

        self.current_dir = Path(__file__).parent

        filepath = self.current_dir / 'data' / 'VNA_Data' / 'fullchain_30dB+30dB_inlineattenuators.s2p'

        super().__init__(filepath = filepath, *args, **kwargs)

        self.tag = f"Setup Cabling (60 dB)"

        spline_s21 = UnivariateSpline(self.f, super().s21_dB, s=1e2)

        self.attenuation = spline_s21(self.f)

        spline_gd = UnivariateSpline(self.f, self.group_delay[:, 1, 0], s=1e2)

        self.gd = spline_gd(self.f)

    @cached_property
    def s21(self):
        spline_s21 = UnivariateSpline(self.f, super().s21, s=1e2)
        attenuation = spline_s21(self.f)
        return attenuation
    
    @cached_property
    def s21_dB(self):
        spline_s21_dB = UnivariateSpline(self.f, super().s21_dB, s=1e2)
        attenuation_dB = spline_s21_dB(self.f)
        return attenuation_dB
    
    @cached_property
    def gd(self):
        spline_gd = UnivariateSpline(self.f, super().gd, s=1e2)
        gd = spline_gd(self.f)
        return gd

if __name__ == '__main__':
    setup = Setup_S21()

    fig, ax = plt.subplots()

    # setup.plot_group_delay(ax = ax)
    setup.plot_S21(ax=ax)
    plt.legend()
    plt.show()