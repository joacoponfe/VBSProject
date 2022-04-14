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
from filters import HPFlspk, LPF1, HPF1, LPFt, HPFt, BPFt1, BPFt2, BPFt
#from dtcwt_matlab import dualfilt1, FSfarras, dualtree1D, idualtree, scalogram
from decomposeSTN import decomposeSTN
from peakdetect import peakdetect
from pydsm import audio_weightings
from weightingFilters import A_weight
from plots import plot_response


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

# Transient processing (NLD)
# LPF and HPF splitting
Nt = 500  # Filter order
SidelobeAtten = 40  # Window Side Lobe attenuation
wint = chebwin(Nt + 1, SidelobeAtten)  # Chebyshev Window
# Apply low pass and high pass filters
yt_lp, b_lp = LPFt(yt, Nt, SidelobeAtten, Fs=Fs2)
yt_hp, b_hp = HPFt(yt, Nt, SidelobeAtten, Fs=Fs2)


###### Plot LPFt and HPFt  filter response ########
#w_lp, h_lp = freqz(b_lp, [1.0])
#w_hp, h_hp = freqz(b_hp, [1.0])
#plot_response(Fs2, w_lp, h_lp, "LPF for Transient Content")
#plot_response(Fs2, w_hp, h_hp, "HPF for Transient Content")
#plt.show()
######################################

# Transfer function coefficients
# Virtual Bass NLD polynomials taken from AES Convention paper 8108 -
# "Analytical and Perceptual Evaluation of Nonlinear Devices for Virtual Bass System".
# Authors: Nay Oo and Woo-Seng Gan
# Comments in ALL CAPS show classes of each NLD according to Oo and Gan:
# "Perceptually-Motivated Objective Grading of Nonlinear Processing in Virtual-Bass Systems".
# - GOOD: good bass enhancement with minimum undesirable distortion.
# - BASS-KILLER: suppress bass perception.
# - NOT RECOMMENDED: produce modest bass enhancement together with undesirable distortion.
# - HIGHLY DISTORTED: just produce a highly distorted bass effect.

# Half-wave rectifier (HWR)  - NOT RECOMMENDED
H_HWR = [0.0278, 0.5, 1.8042, 0, -6.0133, 0, 12.2658, 0, -11.8891, 0, 4.3149]
# Full-wave rectifier (FWR) - NOT RECOMMENDED
H_FWR = [0.0555, 0, 3.6083, 0, -12.0265, 0, 24.5316, 0, -23.7783, 0, 8.6298]
# Polynomial Harmonic Synthesizer (PHS)
H_PHS = [-0.2, -1.2, -3.2, 2.8, 4.8, 0, 0, 0, 0, 0, 0]
# Clipper (CLP) - NOT RECOMMENDED
H_CLP = [0, 0.9517, 0, 1.3448, 0, -7.6028, 0, 10.0323, 0, -4.2418, 0]
# Exponential 1  (EXP1)
H_EXP1 = [0.2689, 0.4255, 0.2127, 0.0709, 0.0177, 0.0035, 0, 0, 0, 0, 0]
# Exponential 2 (EXP2) - GOOD
H_EXP2 = [0, 1.582, -0.791, 0.2637, -0.0659, 0.0132, -0.0022, 0.0003, 0, 0, 0]
# Fuzz Exponential 1 (FEXP1) - GOOD
H_FEXP1 = [0, 1.4847, 0, -1.4449, 0, 2.4713, 0, -2.4234, 0, 0.9158, 0]
# Fuzz Exponential 2 (FEXP2) - BASS-KILLER
H_FEXP2 = [0, 0.6178, 0, 0.7255, 0, -0.8994, 0, 0.8918, 0, -0.3369, 0]
# Arc-tangent square root (ATSR) - GOOD
H_ATSR = [0.0001, 2.7494, -1.0206, -1.0943, -0.1141, 0.7023, -0.4382, -0.3744, 0.5317, 0.0997, -0.3682]

h = H_HWR  # Select which transfer function to use
hN = len(h)  # Number of coefficients in transfer function

for n in np.arange(hN):
    if n == 0:
        yt_low_proc = h[n]
    else:
        yt_low_proc = yt_low_proc + h[n] * (yt_lp ** n)

h = H_FEXP1
hN = len(h)
for n in np.arange(hN):
    if n == 0:
        yt_low_proc_2 = h[n]
    else:
        yt_low_proc_2 = yt_low_proc_2 + h[n] * (yt_low_proc ** n)

# Apply band pass filter (aplico solo uno, no dos como hace Moliner que en realidad es un LP y después un HP)
NBPF1 = 200
#NBPF2 = 100
yt_low_proc_bpf, b_bp = BPFt(yt_low_proc_2, N=NBPF1, Fs=Fs2)

# PLOT RESPONSE OF BPF #
#w_bp, h_bp = freqz(b_bp, [1.0])
#plot_response(Fs2, w_bp, h_bp, "BPF for Transient Content")
#plt.show()
# END PLOT RESPONSE #

#yt_low_proc_bpf = BPFt2(yt_low_proc_bpf, N=NBPF2, Fs=Fs2)

# Delay adjustment
#delay_low = np.round((NBPF1+NBPF2)/2)
delay_low = int(NBPF1/2)
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
#dB_low = np.asarray([])
dB_low = []
#dB_proc = np.asarray([])
dB_proc = []
while pin < pend:
    #iii = iii + 1
    grain_low = yt_low_padded[pin+1:pin+N_gain]
    filtered_grain_low = A_weight(grain_low, Fs2)
    power_low = np.sum(filtered_grain_low ** 2) / N_gain
    #dB_low[iii] = 10 * np.log10(power_low)
    #np.append(dB_low, 10 * np.log10(power_low))
    dB_low.append(10 * np.log10(power_low))

    grain_proc = yt_low_proc_bpf_padded[pin+1:pin+N_gain]
    filtered_grain_proc = A_weight(grain_proc, Fs2)  # Debería ser un K weighting filter, pero por ahora uso un A.
    power_proc = np.sum(filtered_grain_low ** 2) / N_gain
    #dB_proc[iii] = 10 * np.log10(power_proc)
    #np.append(dB_proc, 10 * np.log10(power_proc))
    dB_proc.append(10 * np.log10(power_proc))

    pin = pin + R_gain

dB_low = np.asarray(dB_low)
dB_proc = np.asarray(dB_proc)
difference = dB_low - dB_proc  # Calculate difference (in dB) between low-pass filtered and NLD-processed signals

# Resampling
dB_low_rs = resample(dB_low, Fs2/R_gain, Fs2)
dB_proc_rs = resample(dB_proc, Fs2/R_gain, Fs2)
difference_rs = resample(difference, Fs2/R_gain, Fs2)

zlen = int((N_gain+R_gain)/2)
dB_low_rs = np.concatenate((np.zeros(zlen), dB_low_rs))
dB_proc_rs = np.concatenate((np.zeros(zlen), dB_proc_rs))
difference_rs = np.concatenate((np.zeros(zlen), difference_rs))
np.where(difference_rs < thresh_dB, difference_rs, 0)
difference_rs = savgol_filter(difference_rs, int(N_gain/2+1), 2)  # Savitzky-Golay filter para smoothing (ver si usar otro)
difference_rs = difference_rs[0:len(yt_low_proc_bpf)]

# Apply gain
G = 10 ** (difference_rs/20)
yt_low_proc_gain = G * yt_low_proc_bpf

# Reconstruction
yt_proc = yt_hp + yt_low_proc_gain  # Adding high-pass filtered and processed low-pass signals together

delay_transients = Nt/2 + delay_low

# End of transient processing with NLD #









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
