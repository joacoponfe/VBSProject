import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import pywt
plt.rcParams["font.family"] = "Calibri"


def plot_response(fs, w, h, title):
    """Utility function to plot filter response functions.
    INPUTS:
        fs: sampling frequency (Hz)
        w: frequencies at which h was computed.
        h: frequency response, as complex numbers.
        title: plot title (string)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)))
    ax.set_ylim(-40, 5)
    ax.set_xlim(0, 0.5*fs)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title(title)
    plt.show()


def plot_spectrogram(x, fs, nWin, win, noverlap):
    """Utility function to plot STFT spectrograms."""
    # f, t, Sxx = spectrogram(x, fs,)
    # plt.pcolormesh(t, f, Sxx, shading='gouraud')
    #plt.figure()
    plt.specgram(x, NFFT=nWin, Fs=fs, window=win, noverlap=noverlap, mode='magnitude', scale='dB', cmap='Greys',
                 vmin=-180, vmax=0)
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Amplitud [dB]')
    plt.show()


def plot_harmonics(x, fs, N):
    """Utility function to plot harmonic spectrum."""
    #x_fft = np.abs(rfft(x, N))
    x_fft = 20 * np.log10(np.abs(rfft(x, N))/np.max(np.abs(rfft(x, N))))
    freq = rfftfreq(N, 1/fs)
    plt.plot(freq, x_fft, 'k')
    plt.ylabel('Magnitud [dB]')
    plt.xlabel('Frecuencia [Hz]')
    plt.show()


def plot_frame(X, m, fs, nWin, plot='mag'):
    """ Todo chequear si hay que transponer X!
    Utility function to plot specific frame of STFT spectrogram.
    Resulting plot is in frequency domain.
    INPUT:
    X: STFT spectrogram, X(m,k).
    m: temporal frame index
    win: analysis window
    nWin: analysis window length
    nHop: hop size length
    plot ('mag' or 'phase'): determines whether to plot magnitude or phase spectrum
    """
    # frame = X[m, :]
    frame = X[m, :] / nWin
    if plot == 'mag':
        y = 20*np.log10(np.abs(frame))
    elif plot == 'phase':
        y = np.angle(frame)
    freqs = np.arange(0.0, fs/2 + 1, fs/nWin)
    plt.plot(freqs, y, 'k')
    plt.ylabel('Amplitud [dB]')
    plt.xlabel('Frecuencia [Hz]')
    plt.show()


def plot_audio(x, fs, title='', color='k', label='', linestyle='-', linewidth=0.8):
    """Function to plot audio in the time domain.
    Inputs:
        x: audio data (array)
        fs: sampling frequency (float)
        title: title of the plot (string)
    """
    #plt.figure()
    t = np.linspace(0, len(x)/fs, len(x))
    plt.title(title)
    plt.plot(t, x, color=color, label=label, linestyle=linestyle, linewidth=linewidth)
    #plt.grid()
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.show()


def get_wavelet(wavelet, level=5):
    w = pywt.Wavelet(wavelet)
    if wavelet not in ['bior6.8', 'rbio6.8']:
        [phi, psi, x] = w.wavefun(level=level)
    else:
        [phi, psi, phi_r, psi_r, x] = w.wavefun(level=level)
    return phi, psi, x


def plot_wavelets(wavelets):
    plt.figure()
    i = 1
    num = len(wavelets)
    for wavelet in wavelets:
        phi, psi, x = get_wavelet(wavelet)
        plt.subplot(int(num/2), int(num/3), i)
        plt.plot(x, psi, 'k')
        plt.title(wavelet)
        i += 1
    plt.tight_layout()


def plot_TEM(yt_lp_delay, yt_low_proc_gain, yt_low_proc_gain_matched, envelope_ref, Fs, xlim):
    """ Function for plotting result of temporal envelope matching in NLD (pre and post)."""
    plt.subplot(3, 1, 1)
    plot_audio(yt_lp_delay, Fs, 'yt_lp_delay')
    plot_audio(envelope_ref, Fs, color='b', linestyle='--')
    plt.xlim(xlim)
    plt.subplot(3, 1, 2)
    plot_audio(yt_low_proc_gain, Fs, 'yt_low_proc_gain')
    plot_audio(envelope_ref, Fs, color='b', linestyle='--')
    plt.xlim(xlim)
    plt.subplot(3, 1, 3)
    plot_audio(yt_low_proc_gain_matched, Fs, 'yt_low_proc_gain_matched')
    plot_audio(envelope_ref, Fs, color='b', linestyle='--')
    plt.xlim(xlim)

#def plot_F0detection(r, )
#    """Plots spectrogram of specific frame, peaks detected, borders delimited by f0min and f0max,
#    F0 search regions, """


