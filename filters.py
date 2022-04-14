from scipy.signal import firwin, lfilter, remez
import numpy as np


def LPF1(x, N=2000, SidelobeAtten=60, Fc=2000, Fs=44100):
    """1st low pass filter.
    From Moliner et al. (2020)
    INPUT
        x: input signal
        N: filter order (default is 2000).
        SidelobeAtten: side lobe attenuation (default is 60)
        Fc: cutoff frequency (default is 2000)
        Fs: sampling rate (default is 44100)"""
    b = firwin(N+1, Fc/(Fs/2), pass_zero='lowpass', window=('chebwin', SidelobeAtten), scale=True)
    y = lfilter(b, [1.0], x)
    return y


def HPF1(x, N=2000, SidelobeAtten=60, Fc=2000, Fs=44100):
    """1st high pass filter.
    From Moliner et al. (2020)
    INPUT
        x: input signal
        N: filter order (default is 2000).
        SidelobeAtten: side lobe attenuation (default is 60)
        Fc: cutoff frequency (default is 2000)
        Fs: sampling rate (default is 44100)"""
    b = firwin(N+1, Fc/(Fs/2), pass_zero='highpass', window=('chebwin', SidelobeAtten), scale=True)
    y = lfilter(b, [1.0], x)
    return y


def HPFlspk(x, N=3000, SidelobeAtten=80, Fc=150, Fs=44100):
    """Loudspeaker simulation high pass filter.
    From Moliner et al. (2020)
    INPUT
        x: input signal
        N: filter order (default is 3000).
        SidelobeAtten: side lobe attenuation (default is 80)
        Fc: cutoff frequency (default is 150)
        Fs: sampling rate (default is 44100)"""
    # Design filter
    b = firwin(N+1, Fc/(Fs/2), pass_zero='highpass', window=('chebwin', SidelobeAtten), scale=True)
    # Apply filter
    y = lfilter(b, [1.0], x)
    return y


def LPFt(x, N=500, SidelobeAtten=40, Fc=150, Fs=44100):
    """Low pass filter for transient processing.
    From Moliner et al. (2020)
    INPUT
        x: input signal
        N: filter order (default is 500)
        SidelobeAtten: side lobe attenuation (default is 40)
        Fc: cutoff frequency (default is 150)
        Fs: sampling rate (default is 44100)"""
    # Design filter
    b = firwin(N+1, Fc/(Fs/2), pass_zero='lowpass', window=('chebwin', SidelobeAtten), scale=True)
    # Apply filter
    y = lfilter(b, [1.0], x)
    return y, b


def HPFt(x, N=500, SidelobeAtten=40, Fc=150, Fs=44100):
    """High pass filter for transient processing.
    From Moliner et al. (2020)
    INPUT
        x: input signal
        N: filter order (default is 500)
        Fc: cutoff frequency (default is 150)
        Fs: sampling rate (default is 44100)"""
    # Design filter
    b = firwin(N+1, Fc/(Fs/2), pass_zero='highpass', window=('chebwin', SidelobeAtten), scale=True)
    # Apply filter
    y = lfilter(b, [1.0], x)
    return y, b


def BPFt1(x, N=200, Fc=150, Fs=44100, Wpass=1, Wstop=1, dens=20):
    """Band pass filter 1 for transient processing."
    From Moliner et al. (2020)
    INPUT
        x: input signal
        N: filter order (default is 200)
        Fc: cutoff frequency (default is 150 Hz).
        Fs: sampling rate (default is 44100).
        Wpass: Passband Weight (default is 1).
        Wstop: Stopband Weight (default is 1).
        dens: Density factor (default is 20)."""
    # Array of band edges
    Fpass = Fc+Fc/4
    Fstop = 2*Fc/4
    trans_width = 50
    #bands = np.asarray([0, Fstop - trans_width, Fstop, Fpass, Fpass + trans_width, Fs/2])/(Fs/2) # Sacamos el /(Fs/s) que hacia quilombo y devolvía NANs
    bands = np.asarray([0, Fstop - trans_width, Fstop, Fpass, Fpass + trans_width, Fs / 2])
    gain = np.asarray([0, 1, 0])
    # Array of weights
    W = np.asarray([Wstop, Wpass])
    # Design filter
    b = remez(N+1, bands=bands, desired=gain, weight=None, type='bandpass', grid_density=dens, fs=Fs)
    # Apply filter
    y = lfilter(b, [1.0], x)
    return y, b


def BPFt2(x, N=100, Fc=150, Fs=44100, Wpass=1, Wstop=0.001, dens=20):
    """Band pass filter 2 for transient processing."
    From Moliner et al. (2020)
    INPUT
        x: input signal
        N: filter order (default is 100)
        Fc: cutoff frequency (default is 150 Hz).
        Fs: sampling rate (default is 44100).
        Wpass: Passband Weight (default is 1).
        Wstop: Stopband Weight (default is 0.001).
        dens: Density factor (default is 20)."""
    # Array of band edges
    Fpass = 2*Fc
    Fstop = 5*Fc
    trans_width = 50
    bands = np.asarray([0, Fpass-trans_width, Fpass, Fstop, Fstop+trans_width, Fs/2])
    gain = [0, 1, 0]
    # Array of weights
    W = [Wstop, Wpass]
    # Design filter
    b = remez(N+1, bands=bands, desired=gain, weight=None, type='bandpass', grid_density=dens, fs=Fs)
    # Apply filter
    y = lfilter(b, [1.0], x)
    return y, b


def BPFt(x, N=200, Fc=150, Fs=44100, Wpass=1, Wstop=1, dens=20):
    """Band pass filter for transient processing. Custom."
    Adapted from Moliner et al. (2020)
    INPUT
        x: input signal
        N: filter order (default is 200)
        Fc: cutoff frequency (default is 150 Hz).
        Fs: sampling rate (default is 44100).
        Wpass: Passband Weight (default is 1).
        Wstop: Stopband Weight (default is 1).
        dens: Density factor (default is 20)."""
    # Array of band edges
    Fpass = 4*Fc
    Fstop = Fc
    trans_width = 50
    #bands = np.asarray([0, Fstop - trans_width, Fstop, Fpass, Fpass + trans_width, Fs/2])/(Fs/2) # Sacamos el /(Fs/s) que hacia quilombo y devolvía NANs
    bands = np.asarray([0, Fstop - trans_width, Fstop, Fpass, Fpass + trans_width, Fs / 2])
    gain = np.asarray([0, 1, 0])
    # Array of weights
    W = np.asarray([Wstop, Wpass])
    # Design filter
    b = remez(N+1, bands=bands, desired=gain, weight=None, type='bandpass', grid_density=dens, fs=Fs)
    # Apply filter
    y = lfilter(b, [1.0], x)
    return y, b
