import numpy as np
from scipy.signal import stft
from transientness import transientness


def decomposeSTN(x, S, nWin, nHop, Fs):
    """Fuzzy separation of tonal, transient and noise components.
    Adapted from Moliner et al. (2020). This is turn was adapted from Damskagg and Valimaki (2017)."""
    NFFT = nWin
    win = 'hann'
    nHopA = round(nHop/S)  # Analysis hop size
    OLF = nWin/nHopA  # Analysis overlap factor

    # Compute STFT
    nOL = nWin-nHop  # Number of points to overlap
    X = stft(x, Fs, win, nWin, nOL)

    # Compute transientness (ESTUDIAR ESTO)
    filter_length_t = 600e-3  # in ms
    filter_length_f = 180     # in Hz
    nMedianH = round(filter_length_t*Fs/nHopA)
    nMedianV = round(filter_length_f*NFFT/Fs)

    Rt = transientness(X[2], nMedianH, nMedianV)
    Rs = 1-Rt
    Rn = 1-np.sqrt(abs(Rt-Rs))
    Rt = Rt-Rn/2
    Rs = Rs-Rn/2

    Xs = X*Rs   # Tonal spectrum
    Xt = X*Rt   # Transient spectrum
    Xn = X*Rn   # Noise spectrum
    return X, Xs, Xt, Xn, Rs, Rt, Rn
