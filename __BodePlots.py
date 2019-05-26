import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from pade import pade

np.set_printoptions(threshold=np.inf)


def add_timedelay(tf_num, tf_den, td, **kwargs):
    """ Add a time delay function to a given transfer function
    
    Due to the lack of a proper time delay implementation in `scipy.signal`, this function uses
    the Padé's Approximation of a specified order (slightly modified form of the `pade` function from
    `control.delay`)
        
    param: tf_num, tf_den: respective numerator and denominator of transfer function only
           td: time delay parameter -> 'a' in exp(-as)
           **kwargs: supply additional information to `pade` function -> n: order of Padé terms used
               |_ ! (optional) -> if not specified, n=1 is used !
    
    return: tuple containing numerators and denominators of the combined transfer function
    """

    pade_num, pade_den = pade(td, **kwargs)
    ptf_num = np.poly1d(tf_num)
    ptf_den = np.poly1d(tf_den)
    ppade_num = np.poly1d(pade_num)
    ppade_den = np.poly1d(pade_den)
    num = ptf_num * ppade_num
    den = ptf_den * ppade_den
    return (num.c, den.c)


def bodefy(num=[1], den=[1], td=None, **kwargs):
    """ Create Bode parameters from Transfer Function
    
    If num is omitted while den is provided, it is assumed that the numerator is 1.
        |_ ! den needs to be passed as a keyword argument in this case !
    If den is omitted, it is assumed to be 1 (same as having no denominators)
        |_ ! No need to pass num as a keyword argument in this case !
        
    param: num: numerator of transfer function
           den: denominator of transfer function
           td: time delay amount
           **kwargs = [,n: order of Padé's terms]
        
    return: A tuple containing the frequency, magnitude, and phase of the the given 
            transfer function -> (w, mag, phase)
    """

    freq_range = np.linspace(0.01, 100, num=10 ** 6)
    if td:
        padefy = add_timedelay(num, den, td, **kwargs)
        print(
            f"Time delay uses Padé's Approximation of order: n = {kwargs.get('n', 1)}"
        )
        sys = signal.TransferFunction(*padefy)
    else:
        sys = signal.TransferFunction(num, den)
    print(sys)
    return signal.bode(sys, freq_range)


def bode_plot(filename, tf_params, show=True, savefig=True):
    """ Plot Bode plots of given transfer function"""

    w, mag, phase = tf_params
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f"{filename.split('.')[0]} Bode Plots", fontsize=16)
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
        plt.savefig(filename, dpi=300)
    if show:
        plt.show()


tau = 1  # test

# tf1 = bodefy([tau, 1], [tau, 0])
# bode_plot('T1Q1a.png', tf1)

# tf2 = bodefy([tau, 1])
# bode_plot('T1Q1b.png', tf2)

# tf3 = bodefy([1], [10, 1], td=1, n=2)
# bode_plot('T1Q2.png', tf3, savefig=False)

# tf4 = bodefy([1], [1, 10.1, 1, 0])
# bode_plot('T1Q3.png', tf4, savefig=False)

tf5 = bodefy(den=[10, 1], td=1, n=2)
bode_plot("T1Q3.png", tf5, savefig=False)
