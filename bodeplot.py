from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from pade import pade


class BodePlot:
    """ Main bode construct
    param: numerator -> list of coefficients for numerator terms (descending power order) [optional]
           denominator -> list of coefficients for denominator terms (descending power order) [optional]
           time_delay -> as in exp(-as), where a == time delay factor [optional]
           **kwargs -> dict of optional parameters supplied to pade function // TODO: extras for plot function
    """

    def __init__(self, numerator=[1], denominator=[1], time_delay=None, **kwargs):
        self.num = numerator
        self.den = denominator
        self.td = time_delay
        self.sys = None
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.bodefy()

    @lru_cache(maxsize=2)
    def bodefy(self):
        """ Create Bode parameters from Transfer Function

        return: A tuple containing the frequency, magnitude, and phase of the given 
                transfer function -> (w, mag, phase)
        """

        freq_range = np.linspace(0.01, 100, num=10 ** 6)
        if self.td:
            padefy = self.add_timedelay()
            print(
                f"Time delay uses Padé's Approximation of order: n = {getattr(self, 'n', 1)}"
            )
            self.sys = signal.TransferFunction(*padefy)
        else:
            self.sys = signal.TransferFunction(self.num, self.den)
        print('\n', self.sys)
        return signal.bode(self.sys, freq_range)

    
    @lru_cache(maxsize=2)
    def add_timedelay(self):
        """ Add a time delay function to a given transfer function

        return: tuple containing numerators and denominators of the combined transfer function
        """

        pade_num, pade_den = pade(self.td, getattr(self, "n", 1))
        ptf_num = np.poly1d(self.num)
        ptf_den = np.poly1d(self.den)
        ppade_num = np.poly1d(pade_num)
        ppade_den = np.poly1d(pade_den)
        num = ptf_num * ppade_num
        den = ptf_den * ppade_den
        return (num.c, den.c)

    def plot(self, filename, show=True, savefig=True):
        """ Plot Bode plots of given transfer function"""

        w, mag, phase = self.bodefy()
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(f"{filename} Bode Plots", fontsize=16)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        # subplot 1 -> Bode magnitude plot
        ax1.semilogx(w, mag)
        ax1.grid(which="major", axis="both", linewidth=2)
        ax1.grid(which="minor", axis="x", linewidth=0.5)
        ax1.set_xlim(0.01, 100)
        ax1.set_ylim(-40, 40)
        ax1.set_xlabel("Frequency [rad/s]")
        ax1.set_ylabel("Gain [dB]")

        # subplot 2 -> Bode phase plot
        ax2.semilogx(w, phase)
        ax2.grid(which="major", axis="both", linewidth=2)
        ax2.grid(which="minor", axis="x", linewidth=0.5)
        ax2.set_xlim(0.01, 100)
        ax2.set_ylim(-180, 90)
        major_ticks = np.arange(-180, 91, 30)
        ax2.set_yticks(major_ticks)
        ax2.set_xlabel("Frequency [rad/s]")
        ax2.set_ylabel("Phase [degrees]")

        if savefig:
            plt.savefig(f"{filename}.png", dpi=300)
        if show:
            plt.show()

    def __repr__(self):
        return f"{self.sys}"
