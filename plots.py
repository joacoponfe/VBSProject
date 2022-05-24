import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from scipy.fft import fft, fftfreq, rfft, rfftfreq
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
        y = np.abs(frame)
    elif plot == 'phase':
        y = np.angle(frame)
    freqs = np.arange(0.0, fs/2 + 1, fs/nWin)
    plt.plot(freqs, y, 'k')
    plt.ylabel('Amplitud')
    plt.xlabel('Frecuencia [Hz]')
    plt.show()


#def plot_bin(X, k, fs, win, nWin, nHop, plot='mag'):


def plot_audio(x, fs, title=''):
    """Function to plot audio in the time domain.
    Inputs:
        x: audio data (array)
        fs: sampling frequency (float)
        title: title of the plot (string)
    """
    #plt.figure()
    t = np.linspace(0, len(x)/fs, len(x))
    plt.title(title)
    plt.plot(t, x, 'k')
    #plt.grid()
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.show()
