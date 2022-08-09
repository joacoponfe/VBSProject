import numpy as np
from audioRead import audioRead
from audioWrite import audioWrite
from peakdetect import peakdetect
from decomposeSTN import decomposeSTN
import scipy.signal as ss
from scipy.signal import freqz, remez, hilbert
from scipy.signal.windows import hann
import librosa as lib
from stft import stft
from istft import istft
import matplotlib.pyplot as plt
import IPython.display as ipd
from pydsm import audio_weightings
import pywt


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


def envelope_matching(y_ref, y_target, method):
    """ Temporal envelope matching of signal.

    Parameters
    ----------
    y_ref: ndarray
        Signal from which envelope is extracted.
    y_target: ndarray
        Signal to which envelope is applied.
    method: 1, 2, 3, 4
        Which method to apply. Methods 1 - 3 are wrong, 4 is correct.
    Returns
    -------
    y_out: ndarray
        Signal with temporal envelope matching applied.
    envelope: ndarray
        Envelope of input signal.
    """
    # TODO check that dimensions match
    # IDEAS
    # Hacerlo por ventanas temporales?
    # Chequear qué número de puntos usar en la transformada Hilbert

    envelope_ref = np.abs(hilbert(y_ref))
    envelope_target = np.abs(hilbert(y_target))

    if method == 1:
        # MÉTODO 1
        dB_envelope = 10 * np.log10(envelope_ref ** 2)
        dB_y_target = 10 * np.log10(y_target ** 2)
        dB_difference = dB_envelope - dB_y_target
        G = 10 ** (dB_difference / 20)
    elif method == 2:
        # MÉTODO 2 (es lo mismo que lo anterior, pero sin pasar a dB)
        G = abs(envelope_ref / y_target)
    elif method == 3:
        # MÉTODO 3 (calculando un G constante, idea de Guille)
        env_mean = np.mean(envelope_ref)
        y_mean = np.mean(abs(y_target))
        G = env_mean / y_mean
    elif method == 4:
        # MÉTODO 4 (bueno)
        G = envelope_ref / envelope_target

    # Apply envelope to target signal
    y_out = G * y_target

    # Normalize
    #y_out = y_out/max(abs(y_out))

    return y_out, envelope_ref, envelope_target, G


def get_wavelet(wavelet, level=5):
    w = pywt.Wavelet(wavelet)
    if wavelet not in ['bior6.8', 'rbio6.8']:
        [phi, psi, x] = w.wavefun(level=level)
    else:
        [phi, psi, phi_r, psi_r, x] = w.wavefun(level=level)
    return phi, psi, x


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

# ENVELOPE TESTS

#t = np.linspace(0, .2, 44100)  # 1 s, fs = 44100

# Define carrier
#yc = np.sin(2*np.pi*1000*t)

# Define envelope
#A = np.sin(2*np.pi*50*t)+1

# Define function

#y = A*yc

# Extract envelope
#env = np.abs(hilbert(y))

# Test applying
#y2 = env*yc

# Error
#error = (y2-y)
#mse = ((y2 - y)**2).mean(axis=0)
#print(mse)

# Plot
#plt.figure()
#plt.subplot(3, 1, 1)
#plt.plot(t, yc)
#plt.xlim([0, .025])
#plt.subplot(3, 1, 2)
#plt.plot(t, y)
#plt.plot(t, env, color='g')
#plt.xlim([0, .025])
#plt.subplot(3, 1, 3)
#plt.plot(t, y2)
#plt.xlim([0, .025])
#plt.show()

#print('hola')
