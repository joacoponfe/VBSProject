import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, upfirdn, lfilter, firwin, freqz, savgol_filter
from stft import stft
from istft import istft
from librosa import resample
from audioRead import audioRead
from audioWrite import audioWrite
from scipy.signal.windows import hann, chebwin
import pywt
from filters import HPFlspk, LPF1, HPF1, LPFt, HPFt, BPFt1, BPFt2
# from dtcwt_matlab import dualfilt1, FSfarras, dualtree1D, idualtree, scalogram
from decomposeSTN import decomposeSTN
from peakdetect import peakdetect
from pydsm import audio_weightings
from weightingFilters import A_weight
from test import plot_response


# Load input audio
# x, Fs, path, duration, frames, channels = audioRead('audios/classical_mono_ref.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/jazz_mono_ref.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/pop_mono_ref.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/rock_mono_ref.wav')
x, Fs, path, duration, frames, channels = audioRead('audios/pluck.wav')

# Stereo to mono conversion
x = x.transpose()
if len(x) == 2:
    x = 0.5 * (x[0] + x[1])

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

# Downsampling (hacerlo opcional)
Fs2 = 4096
x_lp = resample(x_lp, Fs, Fs2, res_type='polyphase')
# Polyphase filtering to match values from Matlab resample function.

# Fuzzy separation
# STFT parameters
nWin = 256  # Window size
Ra = 32  # Hop size
noverlap = nWin - Ra  # Overlap size
S = 1  # Time-Scale Modification Factor (TSM)
win = hann(nWin)

X, Xs, Xt, Xn, Rs, Rt, Rn = decomposeSTN(x_lp, S, nWin, Ra, Fs2)

# ISTFT
yt = istft(Xt, Ra, win, win)
yn = istft(Xn, Ra, win, win)
ys = istft(Xs, Ra, win, win)

# PARA TESTEAR LA SEPARACION
# audioWrite('audios/pluck_transient_31_03_fuzzy_Python.wav', yt, Fs2)
# audioWrite('audios/pluck_tonal_31_03_fuzzy_Python.wav', ys, Fs2)
# audioWrite('audios/pluck_noise_31_03_fuzzy_Python.wav', yn, Fs2)

# Transient processing
# LPF and HPF splitting
Nt = 500  # Filter order
SidelobeAtten = 40  # Window Side Lobe attenuation
wint = chebwin(Nt + 1, SidelobeAtten)  # Chebyshev Window
# Apply low pass and high pass filters
yt_lp, b_lp = LPFt(yt, Nt, SidelobeAtten, Fs=Fs2)
yt_hp, b_hp = HPFt(yt, Nt, SidelobeAtten, Fs=Fs2)



###### Check filter response ######## LOW PASS Y HIGH PASS OK, PERO LOS BANDPASS SALEN COMO NAN :(
w_lp, h_lp = freqz(b_lp, [1.0])
w_hp, h_hp = freqz(b_hp, [1.0])
# w_bp1, h_bp1 = freqz(b_hp1, [1.0])
# w_bp2, h_bp2 = freqz(b_hp2, [1.0])
# plt.figure()
# plt.plot(w_lp, abs(h_lp))
# plt.figure()
# plt.plot(w_hp, abs(h_hp))
# plt.figure()
# plt.plot(w_bp1, abs(h_bp1))
# plt.figure()
# plt.plot(w_bp2, abs(h_bp2))

# w_bp1, h_bp1 = freqz(b_bp1, [1.0])
# w_bp2, h_bp2 = freqz(b_bp2, [1.0])
# plt.figure()
# plot_response(Fs2, w_bp1, h_bp1, "BPF 1")
# plt.figure()
# plot_response(Fs2, w_bp2, h_bp2, "BPF 2")
# plt.show()

# plt.show()
######################################

# Transfer function coefficients
H_hwr = [0.0278, 0.5, 1.8042, 0, -6.0133, 0, 12.2658, 0, -11.8891, 0, 4.3149]
H_fexp1 = [0, 1.4847, 0, -1.4449, 0, 2.4713, 0, -2.4234, 0, 0.9158, 0]
h = H_hwr  # Select which transfer function to use
hN = len(h)  # Number of coefficients in transfer function

for n in np.arange(hN):
    if n == 0:
        yt_low_proc = h[n]
    else:
        yt_low_proc = yt_low_proc + h[n] * (yt_lp ** n)

h = H_fexp1
hN = len(h)
for n in np.arange(hN):
    if n == 0:
        yt_low_proc_2 = h[n]
    else:
        yt_low_proc_2 = yt_low_proc_2 + h[n] * (yt_low_proc ** n)

# Apply band pass filters
NBPF1 = 200
#NBPF2 = 100
yt_low_proc_bpf = BPFt1(yt_low_proc_2, N=NBPF1, Fs=Fs2)
#yt_low_proc_bpf = BPFt2(yt_low_proc_bpf, N=NBPF2, Fs=Fs2)

# Delay adjustment
#delay_low = np.round((NBPF1+NBPF2)/2)
delay_low = NBPF1/2
yt_hp = np.concatenate((np.zeros(delay_low), yt_hp))
if len(yt_hp) > len(yt_low_proc_bpf):
    yt_hp = yt_hp[0:len(yt_low_proc_bpf)]
else:
    yt_low_proc_bpf = yt_low_proc_bpf[0:len(yt_hp)]

# Gain calculation
N_gain = 512
R_gain = 64
pend = len(yt_lp)
yt_low_padded = np.concatenate((yt_lp, np.zeros(N_gain)))
yt_low_proc_bpf_padded = np.concatenate((yt_low_proc_bpf, np.zeros(N_gain)))
pin = 0
#iii = 0
dB_low = np.asarray([])
dB_proc = np.asarray([])
while pin < pend:
    #iii = iii + 1
    grain_low = yt_low_padded[pin+1:pin+N_gain]
    filtered_grain_low = A_weight(grain_low)
    power_low = np.sum(filtered_grain_low ** 2) / N_gain
    #dB_low[iii] = 10 * np.log10(power_low)
    np.append(dB_low, 10 * np.log10(power_low))

    grain_proc = yt_low_proc_bpf_padded[pin+1:pin+N_gain]
    filtered_grain_proc = A_weight(grain_proc)  # Debería ser un K weighting filter, pero por ahora uso un A.
    power_proc = np.sum(filtered_grain_low ** 2) / N_gain
    #dB_proc[iii] = 10 * np.log10(power_proc)
    np.append(dB_proc, 10 * np.log10(power_low))

    pin = pin + R_gain

difference = dB_low - dB_proc

# Resampling
dB_low_rs = resample(dB_low,Fs2,Fs2/R_gain)
dB_proc_rs = resample(dB_proc,Fs2,Fs2/R_gain)
difference_rs = resample(difference,Fs2,Fs2/R_gain)

dB_low_rs = np.concatenate((np.zeros((N_gain+R_gain)/2), dB_low_rs))
dB_proc_rs = np.concatenate((np.zeros((N_gain+R_gain)/2), dB_proc_rs))
difference_rs = np.concatenate((np.zeros((N_gain+R_gain)/2), difference_rs))
np.where(difference_rs < thresh_dB, difference_rs, 0)
difference_rs = savgol_filter(difference_rs, N_gain/2, 2)  # Savitzky-Golay filter para smoothing (ver si usar otro)
difference_rs = difference_rs[0:len(yt_low_proc_bpf)]

# Apply gain
G = 10 ** (difference_rs/20)
yt_low_proc_gain = G * yt_low_proc_bpf

# Reconstruction
yt_proc = yt_hp + yt_low_proc_gain  # Adding high-pass filtered and processed low-pass signals together

delay_transients = Nt/2 + delay_low











# STFT parameters
# win_type = 'hann'
# L_win = 256  # Analysis window length
# Ra = 32  # Analysis hop size


# DT-CWT
# Define number of levels to use
N = len(x_lp)  # Calculate length of signal
if not np.log2(N).is_integer():
    J = int(np.ceil(np.log2(N)))  # Rounds up to the nearest integer
    N2 = 2 ** J  # Define new length as nearest power of two
    x_l = np.append(x_lp, np.zeros(N2 - N))  # Zero pad

# Check that lowest frequency band is < 10 Hz.
lowest_freq = Fs / (2 ** J)
while lowest_freq > 10:
    # More levels/scales are required
    J = J + 1  # Increase level
    lowest_freq = Fs / (2 ** J)  # Recompute lowest frequency
    N2 = 2 ** J
    x_l = np.append(x, np.zeros(N2 - N))

Faf, Fsf = FSfarras()  # 1st stage analysis and synthesis filters
af, sf = dualfilt1()  # Remaining stages anal. and synth. filters
x_coeffs, w_coeffs = dualtree1D(x_l, J, Faf, af)
