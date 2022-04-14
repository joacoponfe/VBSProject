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
