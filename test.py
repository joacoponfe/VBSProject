import numpy as np
from audioRead import audioRead
from audioWrite import audioWrite
from peakdetect import peakdetect
from decomposeSTN import decomposeSTN
import scipy.signal as ss
from scipy.signal import freqz, remez
from scipy.signal.windows import hann
import librosa as lib
from stft import stft
from istft import istft
import matplotlib.pyplot as plt
import IPython.display as ipd
from pydsm import audio_weightings


def plot_response(fs, w, h, title):
    """Utility function to plot response functions"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)))
    ax.set_ylim(-40, 5)
    ax.set_xlim(0, 0.5*fs)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title(title)


def plot_audio(x, fs, title):
    """Function to plot audio in the time domain.
    Inputs:
        x: audio data (array)
        fs: sampling frequency (float)
        title: title of the plot (string)
    """
    t = np.linspace(0, len(x)/fs, len(x))
    plt.figure()
    plt.title(title)
    plt.plot(t, x)
    plt.grid()
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')


def fracdelay(delay, N):
    """Implements a fractional delay filter.
    Based on: https://tomroelandts.com/articles/how-to-create-a-fractional-delay-filter.

    Parameters
    ----------
    delay : float
        Fractional delay (in samples)
    N : int
        Filter order

    Returns
    -------
    h : ndarray
        Fractional delay filter coefficients.
    """
    n = np.arange(N)

    # Compute sinc filter
    h = np.sinc(n - (N - 1) / 2 - delay)

    # Multiply sinc filter by window
    h *= np.blackman(N)

    # Normalize to get unity gain
    h /= np.sum(h)

    return h


def lagrange(delay, N):
    n = np.arange(N)
    h = np.ones(N+1)
    for k in np.arange(N):
        index = np.where(n != k)[0]
        for i in index:
            h[i] = h[i] * (delay - k) / (n[i] - k)
    return h



# #x, Fs, path, duration, frames, channels = audioRead('audios/classical_mono_ref.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/pluck.wav')
# nWin = 2048
# win = hann(nWin)
# nHop = 128
# nOL = nWin-nHop
# X, Xs, Xt, Xn, Rs, Rt, Rn = decomposeSTN(x, 1, nWin, nHop, Fs)
# #X, T = stft(x, win, nHop, nWin)
# #x_resynth = istft(X, nHop, win, win)
#
# #ipd.display(ipd.Audio(data=x, rate=Fs))
# #ipd.display(ipd.Audio(data=x_resynth, rate=Fs))
# #audioWrite('audios/resynth.wav', x_resynth, Fs)
# print('hello')
# #plt.figure(1)
# #plt.imshow(Xs)
#
# #plt.figure(2)
# #plt.imshow(Xt)
#
# #plt.figure(3)
# #plt.imshow(Xn)
#
# #plt.show()
#
# #Probando distintos tipos de STFT
# #X1 = ss.stft(x, Fs, 'hann', nWin, nOL)  # Scipy
# #X2 = lib.stft(x, nWin, nHop, nWin, 'hann')  # Librosa
# #X, T = stft(x, win, nHop, nWin)  # Matlab (Moliner et al.)
#
#
# #exact, exact_peak = peakdetect(abs(x), Fs, -4, Fs/4)
# #print('hello')


# fs = 20000.0         # Sample rate, Hz
# band = [6000, 8000]  # Desired stop band, Hz
# trans_width = 200    # Width of transition from pass band to stop band, Hz
# numtaps = 175        # Size of the FIR filter.
# edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]
# taps = remez(numtaps, edges, [0, 1, 0], Hz=fs)
# w, h = freqz(taps, [1], worN=2000)
# plot_response(fs, w, h, "Band-stop Filter")
# plt.show()

# DT-CWT
# Define number of levels to use
# N = len(x_lp)  # Calculate length of signal
# if not np.log2(N).is_integer():
#     J = int(np.ceil(np.log2(N)))  # Rounds up to the nearest integer
#     N2 = 2 ** J  # Define new length as nearest power of two
#     x_l = np.append(x_lp, np.zeros(N2 - N))  # Zero pad
#
# # Check that lowest frequency band is < 10 Hz.
# lowest_freq = Fs / (2 ** J)
# while lowest_freq > 10:
#     # More levels/scales are required
#     J = J + 1  # Increase level
#     lowest_freq = Fs / (2 ** J)  # Recompute lowest frequency
#     N2 = 2 ** J
#     x_l = np.append(x, np.zeros(N2 - N))
#
# Faf, Fsf = FSfarras()  # 1st stage analysis and synthesis filters
# af, sf = dualfilt1()  # Remaining stages anal. and synth. filters
# x_coeffs, w_coeffs = dualtree1D(x_l, J, Faf, af)
