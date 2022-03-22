from scipy.signal import firwin, lfilter


def LPF1(x, N=2000, SidelobeAtten=60, Fc=2000, Fs=44100):
    """1st low pass filter.
    From Moliner et al. (2020)
    INPUT
        x: input signal
        N: filter order (default is 2000).
        SidelobeAtten: side lobe attenuation (default is 60)
        Fc: cutoff frequency (default is 2000)
        Fs: sampling rate (default is 44100)"""
    b = firwin(N+1, Fc/(Fs/2), pass_zero='lowpass', window=('chebwin', SidelobeAtten), scale='True')
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
    b = firwin(N+1, Fc/(Fs/2), pass_zero='highpass', window=('chebwin', SidelobeAtten), scale='True')
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
    b = firwin(N+1, Fc/(Fs/2), pass_zero='highpass', window=('chebwin', SidelobeAtten), scale='True')
    # Apply filter
    y = lfilter(b, [1.0], x)
    return y
