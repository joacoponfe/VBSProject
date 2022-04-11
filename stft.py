import numpy as np
from scipy.fft import fft
from scipy.fft import fftshift


def stft(x, win, nHop, NFFT):
    """Compute the STFT.
    Input parameters:
        x is the input signal.
        win is the analysis window
        nHop is the analysis hop size
        NFFT is the number of points in each DFT."""
    nWin = len(win)
    L = len(x)

    nFrames = int(np.floor((L-nWin)/nHop+1))
    nBins = int(NFFT/2+1)
    Y = np.zeros((nBins, nFrames)).astype(complex)
    T = np.zeros((1, nFrames))

    pin = 0
    x = np.concatenate((np.zeros(nWin), x, np.zeros(nWin-(L % nHop))))
    for n in np.arange(nFrames):
        grain = x[pin:pin+nWin]*win
        f = fft(fftshift(grain), NFFT)  # Investigar más esto del shift
        Y.transpose()[n] = f[0:nBins]
        T.transpose()[n] = pin+1
        pin = pin+nHop
    return Y, T
