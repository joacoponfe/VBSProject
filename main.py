import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, upfirdn, lfilter, firwin, stft, istft
from librosa import resample
from audioRead import audioRead
import pywt
from filters import HPFlspk, LPF1, HPF1
#from dtcwt_matlab import dualfilt1, FSfarras, dualtree1D, idualtree, scalogram
from decomposeSTN import decomposeSTN

# Load input audio
x, Fs, path, duration, frames, channels = audioRead('audios/classical_mono_ref.wav')
#x, Fs, path, duration, frames, channels = audioRead('audios/jazz_mono_ref.wav')
#x, Fs, path, duration, frames, channels = audioRead('audios/pop_mono_ref.wav')
#x, Fs, path, duration, frames, channels = audioRead('audios/rock_mono_ref.wav')

# Stereo to mono conversion
x = x.transpose()
if len(x) == 2:
    x = 0.5*(x[0]+x[1])

# Define parameters
num_harmonics = 5  # Number of processed harmonics
Fcc = 150  # Cut-off frequency [Hz]
GdB = 8  # NLD gain for transient calibration [dB]
thresh_dB = -70  # Magnitude threshold for peak search [dB]
inharmonicity_tolerance = 0.05  # Tolerance parameter (tau) for harmonic detection [%]
alpha_low = 7  # Lower weighting limit in dB/oct
alpha_high = 2  # Upper weighting limit in dB/oct
freq_window = 55  # Size of the frequency window for harmonic enhancement [Hz]
extra_harmonic_gain = 3  # Extra harmonic gain for tonal calibration

# Splitting and resampling

# High-pass filter
x_hp = HPF1(x, Fs=Fs)

# Low-pass filter
x_lp = LPF1(x, Fs=Fs)

# Downsampling
Fs2 = 4096
x_lp = resample(x_lp, Fs2, Fs)

# Fuzzy separation
# STFT parameters
L_win = 256  # Window size
Ra = 32  # Hop size
noverlap = L_win-Ra  # Overlap size
S = 1  # Time-Scale Modification Factor (TSM)

X, Xs, Xt, Xn, Rs, Rt, Rn = decomposeSTN(x_lp, S, L_win, Ra, Fs2)

# ISTFT
yt = istft(Xt, Fs2, 'hann', L_win, noverlap)
yn = istft(Xn, Fs2, 'hann', L_win, noverlap)
ys = istft(Xs, Fs2, 'hann', L_win, noverlap)

# Phase Vocoder

# STFT parameters
win_type = 'hann'
L_win = 256  # Analysis window length
Ra = 32  # Analysis hop size


# DT-CWT
# Define number of levels to use
N = len(x_lp)  # Calculate length of signal
if not np.log2(N).is_integer():
    J = int(np.ceil(np.log2(N)))  # Rounds up to the nearest integer
    N2 = 2**J  # Define new length as nearest power of two
    x_l = np.append(x_lp, np.zeros(N2-N))  # Zero pad

# Check that lowest frequency band is < 10 Hz.
lowest_freq = Fs/(2**J)
while lowest_freq > 10:
    # More levels/scales are required
    J = J + 1  # Increase level
    lowest_freq = Fs/(2**J)  # Recompute lowest frequency
    N2 = 2**J
    x_l = np.append(x, np.zeros(N2-N))

Faf, Fsf = FSfarras()  # 1st stage analysis and synthesis filters
af, sf = dualfilt1()  # Remaining stages anal. and synth. filters
x_coeffs, w_coeffs = dualtree1D(x_l, J, Faf, af)

