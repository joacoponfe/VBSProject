# HACER
# DESPUES ANTITRANSFORMAR CADA COMPONENTE Y ESCUCHAR
import numpy as np
from scipy.fft import ifft
from scipy.fft import fftshift
from ola_norm_coef import ola_norm_coef


def istft(X, nHop, win, win_analysis):
    """Compute the ISTFT.
    Input parameters:
        X is the input spectrogram (positive frequencies and DC).
        nHop is the hop size
        win is the synthesis window (default is rectangular)
        win_analysis is the analysis window used for computing spectrogram X"""
    nWin = len(win)
    nBins = X.shape[0]
    nFrames = X.shape[1]
    NFFT = (nBins-1)*2

    # Length of output
    L = (nFrames-1)*nHop+nWin
    y = np.zeros(L)

    # OLA normalization
    norm_coef = ola_norm_coef(win_analysis, win, nHop)

    # Compute two-sided spectrogram
    XF = np.zeros((NFFT, nFrames)).astype(complex)
    XF[0:nBins] = X[0:nBins]
    XF[nBins:-1] = np.conj(np.flipud(X[1:-2]))

    # Overlap-add synthesis
    p = 0
    for n in np.arange(nFrames):
        grain = fftshift(np.real(ifft(XF[:, n])))
        grain = grain[0:nWin]*win/norm_coef
        y[p:p+nWin] = y[p:p+nWin] + grain
        p = p + nHop
    return y
