import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.filter_design
from scipy.signal import butter, upfirdn, lfilter, firwin, freqz, savgol_filter
from stft import stft
from istft import istft
from librosa import resample
from librosa.filters import mel, mel_frequencies
from audioRead import audioRead
from audioWrite import audioWrite
from scipy.signal.windows import hann, chebwin, tukey
from scipy.interpolate import interp1d, pchip_interpolate
import pywt
from filters import HPFlspk, LPFlspk, LPF1, HPF1, LPFt, HPFt, BPFt1, BPFt2, BPFt
from decomposeSTN import decomposeSTN
from peakdetect import peakdetect
from pydsm import audio_weightings
from weightingFilters import A_weight
from melbank import compute_mat, hertz_to_bark, bark_to_hertz
import copy
from plots import plot_response, plot_spectrogram, plot_audio, plot_harmonics, plot_frame, plot_TEM, plot_wavelets, plot_PV
from NLD import NLD
from utils import fracdelay, lagrange, envelope_matching, get_wavelet
import datetime
from os.path import basename
import time

# Other settings
np.seterr(invalid='ignore')  # To ignore RuntimeWarnings

# Start timer
tic = time.time()


# Load input audio
# x, Fs, path, duration, frames, channels = audioRead('audios/music/loudnorm/classical_mono_ref.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/music/loudnorm/jazz_mono_ref.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/music/loudnorm/pop_mono_ref.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/music/loudnorm/rock_mono_ref.wav')
x, Fs, path, duration, frames, channels = audioRead('audios/NLD/tone_100Hz_Amp_1.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/multitonal_100_43.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/multitonal_38_43_98.wav')

name = basename(path)[:-4]  # Get basename of audio file without extension (will be used when saving output files)

# Stereo to mono conversion
x = x.transpose()
if len(x) == 2:
    x = 0.5 * (x[0] + x[1])

# Define parameters
num_harmonics = 6  # Number of processed harmonics
#Fcc = 150  # Cut-off frequency [Hz] (MOLINER)
Fcc = 200
#Fcc = 250  # Cut-off frequency [Hz] (Según lo que escribí en el Marco Teórico)
thresh_dB = -70  # Magnitude threshold for peak search [dB]
inharmonicity_tolerance = 0.05  # Tolerance parameter (tau) for harmonic detection [%]
alpha_low = 7  # Lower weighting limit in dB/oct
alpha_high = 2  # Upper weighting limit in dB/oct
freq_window_Hz = 55  # Size of the frequency window for harmonic enhancement [Hz]
#freq_window_Hz = 100
#freq_window_Hz = 60
extra_harmonic_gain = 3  # Extra harmonic gain for tonal calibration
sep_method = 'Median'  # Select component separation method ('MCA' or 'Median')

# STFT parameters
# nWin = 2756  # Window size (2756 para mantener la relación de 16 entre 4096/256 y 44100/2756)
# nWin = 1024
nWin = 256
nHop = 32  # Hop size
noverlap = nWin - nHop  # Overlap size
S = 1  # Time-Scale Modification Factor (TSM)
win = hann(nWin)

# Splitting and resampling

N1 = 2000  # Filter order for 1st HPF and LPF
Fc1 = 2000  # Cut-off frequency for 1st HPF and LPF

# High-pass filter
x_hp, b_hp1 = HPF1(x, N=N1, Fc=Fc1, Fs=Fs)

# Low-pass filter
x_lp, b_lp1 = LPF1(x, N=N1, Fc=Fc1, Fs=Fs)
# x_lp = x  # sólo para fuzzy tests

#############################################
# Plot LPF1 and HPF1  filter response
# w_lp1, h_lp1 = freqz(b_lp1, [1.0])
# w_hp1, h_hp1 = freqz(b_hp1, [1.0])
# plot_response(Fs, w_lp1, h_lp1, "LPF1")
# plot_response(Fs, w_hp1, h_hp1, "HPF1")
# plt.show()
#############################################

# Downsampling (hacerlo opcional)
Fs2 = 4096
# Fs2 = Fs
x_lp = resample(x_lp, Fs, Fs2, res_type='polyphase')
# Polyphase filtering to match values from Matlab resample function.

if sep_method == 'Median':  # Median filtering method (Fuzzy) based on Fitzgerald (2010).
    X, Xs, Xt, Xn, Rs, Rt, Rn = decomposeSTN(x_lp, S, nWin, nHop, Fs2)
    # ISTFT
    yt = istft(Xt, nHop, win, win)
    yn = istft(Xn, nHop, win, win)
    ys = istft(Xs, nHop, win, win)
    yfull = istft(X, nHop, win, win)
elif sep_method == 'MCA':  # MCA
    # Save low-passed and downsampled audio for MCA processing externally
    #audioWrite(f'audios/MCA/{name}_lp_4096.wav', x_lp, Fs2)
    # Load separated audios (MCA outputs)
    yt = audioRead(f'audios/MCA/{name}_lp_4096_MCA_transient_bior6.8.wav')[0]  # Transient
    ys = audioRead(f'audios/MCA/{name}_lp_4096_MCA_tonal_bior6.8.wav')[0]  # Tonal
    # Calculate STFT of tonal signal for PV
    Xs, T = stft(ys, win, nHop, nWin)
    # Make loaded audios the same length as Median separation method, to match dimensions for the rest of the code.
    ys = istft(Xs, nHop, win, win)
    yt = np.concatenate((np.zeros(nWin), yt))
    yt = yt[:len(ys)]

#########################################################################
# Para testear la separación
if sep_method == 'Median':
    # Tomamos desde la muestra nº nWin en adelante, para que se alineen los audios.
    # Después, rellenamos con ceros para que tengan igual longitud que x_lp.
    ys_out = ys[nWin:]
    ys_out = np.concatenate((ys_out, np.zeros(len(x_lp) - len(ys_out))))
    yt_out = yt[nWin:]
    yt_out = np.concatenate((yt_out, np.zeros(len(x_lp) - len(yt_out))))
    yfull_out = yfull[nWin:]
    yfull_out = np.concatenate((yfull_out, np.zeros(len(x_lp)-len(yfull_out))))

# Write output files with alignment to x_lp.
#audioWrite(f'audios/{name}_{sep_method}_tonal.wav', ys_out, Fs2)
#audioWrite(f'audios/{name}_{sep_method}_transient.wav', yt_out, Fs2)
#audioWrite(f'audios/{name}_{sep_method}_full.wav', yfull_out, Fs2)
#audioWrite('audios/museval/fuzzy_MCA/MCA_fuzzy_tonal_stimulusB_2.wav', ys[nWin:], Fs2)
#audioWrite('audios/museval/fuzzy_MCA/fuzzy_noise_stimulusB.wav', yn[nWin:], Fs2)
#audioWrite('audios/museval/fuzzy_MCA/full_MCA_Fuzzy.wav', yfull[nWin:], Fs2)

# Write output files without alignment.
#audioWrite(f'audios/{name}_{sep_method}_tonal.wav', ys, Fs2)
#audioWrite(f'audios/{name}_{sep_method}_transient.wav', yt, Fs2)
#########################################################################

# Transient processing (NLD)
# LPF and HPF splitting
Nt = 500  # Filter order
SidelobeAtten = 40  # Window Side Lobe attenuation
# TODO Warning: for attenuations lower than about 45 dB this window is not suitable for spectral analysis.
#  See whether it should be changed.
wint = chebwin(Nt + 1, SidelobeAtten)  # Chebyshev Window (nunca se usa)
# Apply low pass and high pass filters
yt_lp, b_lp = LPFt(yt, Nt, SidelobeAtten, Fc = Fcc, Fs=Fs2)
yt_hp, b_hp = HPFt(yt, Nt, SidelobeAtten, Fc = Fcc, Fs=Fs2)

################################################################
# Plot LPFt and HPFt  filter response
# w_lp, h_lp = freqz(b_lp, [1.0])
# w_hp, h_hp = freqz(b_hp, [1.0])
# plot_response(Fs2, w_lp, h_lp, "LPF for Transient Content")
# plot_response(Fs2, w_hp, h_hp, "HPF for Transient Content")
# plt.show()
################################################################

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
# Polynomial Harmonic Synthesizer (PHS) - NO CLASSIFICATION
H_PHS = [-0.2, -1.2, -3.2, 2.8, 4.8, 0, 0, 0, 0, 0, 0]
# Clipper (CLP) - NOT RECOMMENDED
H_CLP = [0, 0.9517, 0, 1.3448, 0, -7.6028, 0, 10.0323, 0, -4.2418, 0]
# Exponential 1  (EXP1) - NO CLASSIFICATION
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
        yt_low_proc = h[n]  # For n = 0, yt_lp^0 = 1
    else:
        yt_low_proc = yt_low_proc + h[n] * (yt_lp ** n)

h = H_FEXP1
hN = len(h)
for n in np.arange(hN):
    if n == 0:
        yt_low_proc_3 = h[n]
    else:
        yt_low_proc_3 = yt_low_proc_3 + h[n] * (yt_low_proc ** n)

h = H_ATSR
hN = len(h)
for n in np.arange(hN):
    if n == 0:
        yt_low_proc_2 = h[n]
    else:
        yt_low_proc_2 = yt_low_proc_2 + h[n] * (yt_low_proc_3 ** n)

# Create highly distorted signal (Anchor 1 for Audio Quality test)
# First, we're using one of the NOT RECOMMENDED NLDs
h = H_FWR
hN = len(h)
for n in np.arange(hN):
    if n == 0:
        yt_dist = h[n]
    else:
        yt_dist = yt_dist + h[n] * (yt_lp ** n)

# Apply band pass filter (aplico solo uno, no dos como hace Moliner que en realidad es un LP y después un HP)
# NBPFt = 200
NBPF1 = 300
# NBPF2 = 100
yt_low_proc_bpf, b_bp = BPFt(yt_low_proc_2, N=NBPF1, Fc = Fcc, Fs=Fs2)
yt_dist_bpf, b_bp = BPFt(yt_dist, N=NBPF1, Fc = Fcc, Fs=Fs2)       # Distorted signal (Anchor 1)

################################################################
# PLOT RESPONSE OF BPF FOR TRANSIENT CONTENT
# w_bp, h_bp = freqz(b_bp, [1.0])
# plot_response(Fs2, w_bp, h_bp, "BPF for Transient Content")
# plt.show()
# END PLOT RESPONSE #
# yt_low_proc_bpf = BPFt2(yt_low_proc_bpf, N=NBPF2, Fc = Fcc, Fs=Fs2)
################################################################

# Delay adjustment
# delay_low = np.round((NBPF1+NBPF2)/2)
delay_low = int(NBPF1 / 2)  # delay = Filter order / 2
yt_hp = np.concatenate((np.zeros(delay_low), yt_hp))
if len(yt_hp) > len(yt_low_proc_bpf):
    yt_hp = yt_hp[0:len(yt_low_proc_bpf)]
else:
    yt_low_proc_bpf = yt_low_proc_bpf[0:len(yt_hp)]
    yt_dist_bpf = yt_dist_bpf[0:len(yt_hp)]             # Distorted signal (Anchor 1)

# Gain calculation
N_gain = int(512)
R_gain = int(64)
pend = len(yt_lp)
yt_low_padded = np.concatenate((yt_lp, np.zeros(N_gain)))
yt_low_proc_bpf_padded = np.concatenate((yt_low_proc_bpf, np.zeros(N_gain)))
pin = 0
# iii = 0
# dB_low = np.asarray([])
dB_low = []
# dB_proc = np.asarray([])
dB_proc = []
while pin < pend:
    # iii = iii + 1
    grain_low = yt_low_padded[pin:pin + N_gain]
    filtered_grain_low = A_weight(grain_low, Fs2)
    power_low = np.sum(filtered_grain_low ** 2) / N_gain
    # dB_low[iii] = 10 * np.log10(power_low)
    # np.append(dB_low, 10 * np.log10(power_low))
    dB_low.append(10 * np.log10(power_low))

    grain_proc = yt_low_proc_bpf_padded[pin:pin + N_gain]
    filtered_grain_proc = A_weight(grain_proc, Fs2)  # Debería ser un K weighting filter, pero por ahora uso un A.
    power_proc = np.sum(filtered_grain_proc ** 2) / N_gain
    # dB_proc[iii] = 10 * np.log10(power_proc)
    # np.append(dB_proc, 10 * np.log10(power_proc))
    dB_proc.append(10 * np.log10(power_proc))

    pin = pin + R_gain

dB_low = np.asarray(dB_low)
dB_proc = np.asarray(dB_proc)
difference = dB_low - dB_proc  # Calculate difference (in dB) between low-pass filtered and NLD-processed signals

# Resampling (back to Fs2)
dB_low_rs = resample(dB_low, Fs2 / R_gain, Fs2)
dB_proc_rs = resample(dB_proc, Fs2 / R_gain, Fs2)
difference_rs = resample(difference, Fs2 / R_gain, Fs2)

zlen = int((N_gain + R_gain) / 2)
dB_low_rs = np.concatenate((np.zeros(zlen), dB_low_rs))
dB_proc_rs = np.concatenate((np.zeros(zlen), dB_proc_rs))
difference_rs = np.concatenate((np.zeros(zlen), difference_rs))
difference_rs[np.where(dB_low_rs < thresh_dB)[0]] = 0
#difference_rs = savgol_filter(difference_rs, int(N_gain / 2 + 1),2)  # Savitzky-Golay filter for smoothing (ver si usar otro)
difference_rs = difference_rs[0:len(yt_low_proc_bpf)]

# Apply gain
# Revisar esto, no está aplicando tanta ganancia como el código de Moliner de Matlab
G = 10 ** (difference_rs / 20)
yt_low_proc_gain = G * yt_low_proc_bpf
yt_low_proc_gain = yt_low_proc_bpf  # Uncomment to bypass old gain method (to see whether TEM works better on its own)

# Temporal envelope matching
yt_lp_delay = np.concatenate((np.zeros(delay_low), yt_lp))  # Add delay caused by BFPt to low-passed signal
if len(yt_lp_delay) > len(yt_low_proc_gain):
    yt_lp_delay = yt_lp_delay[0:len(yt_low_proc_gain)]
else:
    yt_low_proc_gain = yt_low_proc_gain[0:len(yt_lp_delay)]

# Calculate envelope of yt_lp signal and apply it to yt_low_proc_gain
yt_low_proc_gain_matched, envelope_ref, envelope_target, gain = envelope_matching(yt_lp_delay, yt_low_proc_gain, method=4)

# Apply BAD Temporal Envelope Matching (with clipping distortion) to Anchor 1 signal
yt_dist_bpf_matched, envelope_ref_2, envelope_target_2, gain_2 = envelope_matching(yt_lp_delay, yt_dist_bpf, method=1)

# Reconstruction
yt_proc = yt_hp + yt_low_proc_gain_matched  # Adding high-pass filtered and processed low-passed signals together
#yt_proc = yt_hp + yt_low_proc_gain  # Uncomment to bypass Temporal Envelope Matching

yt_dist_proc = yt_hp + yt_dist_bpf_matched  # Same, but for highly distorted (Anchor 1) signal

delay_transients = Nt / 2 + delay_low  # Delay for transient signal: N(LPFt)/2 + N(BPFt)/2

# End of transient processing with NLD #

# audioWrite('audios/classical_NLD_processed_28-04.wav', yt_proc, Fs2)

# Harmonic processing (PV) #

# Harmonic detection variables

# Initialization
nBins, nFrames = np.shape(Xs)

detected_peaks = [[] for _ in np.arange(nFrames)]
detected_peaks_values = [[] for _ in np.arange(nFrames)]

f0found = np.zeros(nFrames).astype('int')  # Which region f0 is found, ranges from 0 to 3.
# 0: not in any region; 1: first region, 2: second region, 3: third region.
detected_f0 = np.zeros(
    (3, nFrames))  # f0 frequency bins [nBin], and where in any of the 3 possible regions it is located.
detected_f0_values = np.zeros((3, nFrames))  # f0 magnitude values [dB]
detected_harmonics = np.zeros((num_harmonics, nFrames))  # Harmonic frequency bins [nBin]
detected_harmonics_values = np.zeros((num_harmonics, nFrames))  # Harmonic magnitude values [dB]
fixed_low = np.zeros((nBins, nFrames)) * (-1) * np.inf  # Weighting envelope lower limit
fixed_high = np.zeros((nBins, nFrames)) * np.inf  # Weighting envelope upper limit
accum_phases = np.zeros(nBins)  # Accumulated phases
YL = np.zeros((nBins, nFrames)).astype('complex')  # Synthesized tonal spectrogram

# Get frequency bins corresponding to f0max and f0min (Hz -> bin)
f0max = Fcc * nWin / Fs2  # f0max is set as the cut-off frequency (Fcc)
#f0min = f0max / 4  # f0min is set as f0max/4
f0min = f0max / 8

# Harmonic enhancement variables
freq_window_size = int(np.round(freq_window_Hz * nWin / Fs2))  # Size of frequency window in bins
if freq_window_size % 2 == 0:  # If window is even-sized
    freq_window_size = freq_window_size + 1  # Becomes odd-sized
freq_window = tukey(freq_window_size, 0.75)  # Region of influence window (w_ROI)
for i in np.argwhere(freq_window == 0):
    freq_window[i] = 1e-3

# Harmonic weighting variables
target_weights_timbre = np.zeros((nBins, nFrames))  # Weighting values
numBands = 7  # Number of bark scale bands
# Range of frequencies where the weighting will be applied (in Hz)
# These specify the center frequencies (not edges) of the lowest and highest filters.
range_min = Fcc / 8
#range_min = Fcc / 4
range_max = Fcc * num_harmonics
range = [range_min, range_max]

# Filter bank, Bark scale TODO revisar la FB, no coincide con Matlab. Ya está hecho la conversión a Bark y lo de center freqs.
FB, (cf, freqs) = compute_mat(num_bands=numBands, freq_min=range[0], freq_max=range[1], num_fft_bands=nWin,
                              sample_rate=Fs2, scale='bark')

cf = bark_to_hertz(cf)  # Center frequencies [Hz]

FBB = np.zeros((nBins, numBands))
FBB = FB.transpose()[:nBins, :]  # Mel Matrix for multiplying with Xs
FBB = FBB / np.sum(FBB, 0)  # Some kind of normalization
cfbins = np.round(cf * nWin / Fs2)  # Frequency bins corresponding to center frequencies of filterbank.

# Tonal processing
# Step 1: Fundamental and Harmonic detection
for n in np.arange(nFrames):
    #n = 42  # DESPUES SACAR, ES PARA PRUEBAS
    f = Xs[:, n] / nWin  # TODO see why it is normalized by nWin
    r = np.abs(f)  # Magnitude spectrum

    bark = np.dot(FBB.transpose(), r)  # Transform STFT spectrum to Bark spectrum

    # All local maxima in the magnitude spectrum are selected as peaks.
    # Fmin = Fcc / 4
    [exact, exact_peak] = peakdetect(r, Fs2, thresh_dB, range_min)

    # Save in lists
    detected_peaks[n] = exact
    detected_peaks_values[n] = exact_peak

    # Possible improvement: change thresh_dB level depending on the dynamic range of the signal.

    # Create vector of frequency bins from first to last center frequency of bark filters (xq = query points)
    xq = np.arange(cfbins[0], cfbins[-1] + 1)

    # Interpolation of peak position (index) in Bark frequency scale
    interpbark = np.zeros(nBins)
    # fx = interp1d(cfbins, bark, kind='slinear')           # Interpolation function bark = f(cfbins)
    # interpbark_xq = fx(xq)                                # Values of interpolation at query points (xp)
    interpbark_xq = pchip_interpolate(cfbins, bark, xq)
    interpbark[int(xq[0]):int(xq[-1] + 1)] = interpbark_xq  # Place interpolated values into interpbark vector
    interpbark = 20 * np.log10(interpbark)                  # Convert to dB

    # Get locations and magnitudes of detected peaks that are below f0max (candidates for F0).
    locations_fundamental_all = exact[np.where(exact < f0max)[0]]  # Evaluate all locations of detected peaks
    peak_fundamental_all = exact_peak[np.where(exact < f0max)[0]]  # Evaluate all magnitudes of detected peaks

    if np.size(peak_fundamental_all) != 0:     # If peaks have been detected
        # The maximum peak from the set of detected peaks is selected as the
        # fundamental candidate.
        maxim = np.max(peak_fundamental_all)            # Magnitude of maximum / f0 candidate
        loc = np.argmax(peak_fundamental_all)           # Index of maximum / f0 candidate
        maximloc = locations_fundamental_all[loc]       # Frequency bin of maximum / f0 candidate
        alimit = maximloc * inharmonicity_tolerance     # Inharmonicity tolerance limit
        borders = np.array([1 / 3, 1 / 2, 1]) * f0max   # Borders of regions to find f0

        # Evaluates whether the candidate is the true f0 component,
        # or whether it could be the second harmonic of another peak in a lower frequency.

        if maximloc / 3 > f0min:
            f0cmin = (maximloc - alimit) / 3             # Determine lower inharmonicity tolerance limit
            f0cmax = (maximloc + alimit) / 3             # Determine upper inharmonicity tolerance limit
            a = np.min(abs(exact - maximloc / 3))        # Searches within detected peaks for a possible f0 candidate
            s = np.argmin(abs(exact - maximloc / 3))     # Gets index
            if f0cmin < exact[s] < f0cmax:               # If f0 candidate is within region of inharmonicity tolerance
                b = np.where(exact[s] <= borders)[0][0]  # Find below which border the f0 candidate is located.
                # (0 = below first border, 1 = below second border, 2 = below third border)
                detected_f0[b, n] = exact[s]              # Save detected f0 position for this frame and in its region.
                detected_f0_values[b, n] = exact_peak[s]  # Save detected f0 magnitude for this frame and in its region.
                f0found[n] = b + 1               # 1 is added to allow following line to perform f0found[n] == 0 search.

        if f0found[n] == 0 and maximloc / 2 > f0min:   # If no f0 has been found and 1/2 of f0 candidate is above f0min
            f0cmin = (maximloc - alimit) / 2           # Determine lower inharmonicity tolerance limit
            f0cmax = (maximloc + alimit) / 2           # Determine upper inharmonicity tolerance limit
            a = np.min(abs(exact - maximloc / 2))      # Searches within detected peaks for a possible f0 candidate
            s = np.argmin(abs(exact - maximloc / 2))   # Gets index
            if f0cmin < exact[s] < f0cmax:             # If f0 candidate is within region of inharmonicity tolerance
                b = np.where(exact[s] <= borders)[0][0]   # Find below which border the f0 candidate is located.
                detected_f0[b, n] = exact[s]              # Save detected f0 position for this frame and in its region.
                detected_f0_values[b, n] = exact_peak[s]  # Save detected f0 magnitude for this frame and in its region.
                f0found[n] = b + 1               # 1 is added to allow following line to perform f0found[n] == 0 search.

        if f0found[n] == 0:                            # If no f0 has been found
            b = np.where(maximloc <= borders)[0][0]    # Find below which border the f0 candidate is located
            detected_f0[b, n] = maximloc               # Save detected f0 position for this frame and in its region.
            detected_f0_values[b, n] = maxim           # Save detected f0 magnitude for this frame and in its region.
            f0found[n] = b + 1

        # Search all respective harmonics of f0
        # f_k = k * f0

        for h in np.arange(num_harmonics):
            # For the number of harmonics set, search for all respective harmonics of f0,
            # and, if these are within the region determined by the inharmonicity parameter,
            # save as detected harmonics.

            # All harmonics should be located above the cut-off frequency (Fcc / f0max).
            # The order of the harmonics to process depends on which interval f0 is located (Moliner et al., 2020).
            # This is established following Plomp's "rule" that for higher f0's, the first harmonics
            # are more important, whereas for low f0's, harmonics fifth and higher are most important.

            # fk = f2, f3, ..., fk+1 if f0 € [fc/2, fc]    <-->  f0found[n] = 3
            # fk = f3, f4, ..., fk+2 if f0 € [fc/3, fc/2]  <-->  f0found[n] = 2
            # fk = f4, f5, ..., fk+3 if f0 € [fc/4, fc/3]  <-->  f0found[n] = 1

            k = h + 5 - f0found[n]  # Order of harmonic
            # 5 is there because f0found ranges from 1 to 3,
            # and in the latter case we want the second harmonic (not f0) to be 2*f0.
            fk = detected_f0[f0found[n] - 1, n] * k

            a = np.min(abs(exact - fk))     # Position of harmonic candidate
            b = np.argmin(abs(exact - fk))  # Index of harmonic candidate

            alimit = fk * inharmonicity_tolerance   # Inharmonicity tolerance limit

            # If position of possible detected harmonic is within the region
            # of inharmonicity tolerance, save as detected harmonic;
            # otherwise, discard.

            if a < alimit:
                detected_harmonics[h, n] = exact[b]               # Position of harmonic
                detected_harmonics_values[h, n] = exact_peak[b]   # Magnitude of harmonic
            else:
                detected_harmonics[h, n] = 0
                detected_harmonics_values[h, n] = 0

        # Scaling up of the envelope to the value of detected F0 component.
        interpbark = interpbark + (maxim + 10 ** (extra_harmonic_gain / 20)) - interpbark[int(np.round(maximloc))]

    else:
        interpbark = interpbark * (-1) * np.inf

    # Create two constant exponentially decaying curves, ENVlow (fixed_low) and ENVhigh (fixed_high),
    # starting at position of f0max (Fcc) and with slopes alpha_low and alpha_high, respectively.

    start_value = interpbark[int(np.floor(f0max))]
    fixed_low[0:int(np.floor(f0max)), n] = (-1) * np.inf
    fixed_high[0:int(np.floor(f0max)), n] = np.inf
    fixed_low[int(np.floor(f0max)):nBins, n] = start_value - alpha_low * np.arange(
        nBins - np.floor(f0max)) / np.floor(f0max)
    fixed_high[int(np.floor(f0max)):nBins, n] = start_value - alpha_high * np.arange(
        nBins - np.floor(f0max)) / np.floor(f0max)

    target_weights_timbre[:, n] = interpbark  # Assign weighting values.

    fsynth = copy.copy(f)  # Create frequency vector (complex) where synthesized version will be saved.

    # Initialization
    shiftleft = np.zeros(num_harmonics)     # Bin shift left
    shiftright = np.zeros(num_harmonics)    # Bin shift right
    new_freq_bin = np.zeros(num_harmonics)  # Frequency bin for new harmonics
    sel_weight = np.zeros(num_harmonics)    # Selected weight for new harmonics

    if f0found[n] != 0:  # If f0 has been found
        fundamental_exact = detected_f0[f0found[n] - 1, n]  # Exact value of fundamental
        fundamental_bin = np.floor(fundamental_exact)       # Frequency bin where fundamental is located

        # Define limits of region of influence for the shifting.
        # So far it has been defined as a rectangular window of fixed size freq_window_size.
        # This could be a possible area of improvement:
        #   - Different freq_window_size for different frequencies?
        #   - Different type of window?

        delta = np.floor(freq_window_size / 2)
        left = int(fundamental_bin - delta)         # Left extreme of the region of influence
        right = int(fundamental_bin + delta + 1)    # Right extreme of the region of influence

        region = f[left:right]          # Region of influence (ROI)
        region_r = np.abs(region)       # Magnitude
        region_phi = np.angle(region)   # Phase

        for nh in np.arange(num_harmonics):
            beta = nh + 5 - f0found[n]                  # Same criteria for defining k previously
            if detected_harmonics[nh, n] == 0:          # No harmonics where detected, so they will be generated
                newfreq = beta * fundamental_exact      # Exact location of new harmonic
                binshift = newfreq - fundamental_exact  # How much shifting of the fundamental is necessary.

                # binshift refers to a number of samples needed to move the f0 component (fundamental_exact)
                # to the position of the required harmonic (newfreq), but it is most likely not an integer number.
                # Given that we are working with frequency bins (integer numbers), this is a problem.
                # This is the reason why we need to use a fractional delay filter, to interpolate magnitude
                # values for the instants between the discrete sampled values.

                shift = binshift * 2 * np.pi / nWin  # bin shift in rad/s?

                # Fractional Delay Filter using a Lagrange Interpolator.
                # The fractional delay filter is used for a more precise shifting of the frequency bins to the
                # exact target frequency, due to the rounding errors on shifting FFT bins.
                # TODO Not exactly the same as the filter used by Moliner, but close enough.

                orderfracfilter = 4                             # Order of the fractional delay filter
                delay = binshift - np.floor(binshift)           # Delay in samples (non integer)
                h_lagrange = lagrange(delay, orderfracfilter)   # Filter coefficients

                shiftleft[nh] = left + np.floor(binshift)       # Bin value for left extreme of the shifted region
                shiftright[nh] = right + np.floor(binshift)     # Bin value for right extreme of the shifted region

                # Phase unwrapping
                region_phi = np.unwrap(region_phi)  # Unwrap radian phase such that adjacent differences are never
                # greater than pi by adding 2kpi for some integer k.
                region_phi_pad = np.append(region_phi, np.zeros(int(orderfracfilter / 2)))  # Padding to compensate for
                # delay generated by fractional delay filter.
                region_phi_filtered = np.convolve(region_phi_pad, h_lagrange, 'same')   # Filter with Lagrange interp.
                region_phi_filtered = region_phi_filtered[int(orderfracfilter / 2):len(region_phi_filtered + 1)]
                # Remove samples caused by delay.

                # TODO revisar y entender bien esto
                # DESDE ACÁ #
                p0 = accum_phases[round(newfreq) - 1]  # Get accumulated phases for harmonic location
                pu = p0 + nHop * shift  # Phase of next frame?
                region_phi_x = region_phi_filtered + pu  # Accumulated phase is added to phase of filtered region (?)
                accum_phases[int(shiftleft[nh]):int(shiftright[nh])] = np.ones(len(region)) * pu
                # HASTA ACÁ #

                timbre_weight = target_weights_timbre[round(newfreq), n]  # Get harmonic weighting values
                new_freq_bin[nh] = round(newfreq)                         # Harmonic bin

                low_weight = fixed_low[round(newfreq), n]
                high_weight = fixed_high[round(newfreq), n]

                # If timbre weight is not within low and high thresholds, adjust accordingly.
                if timbre_weight < low_weight:
                    sel_weight[nh] = low_weight
                elif timbre_weight > high_weight:
                    sel_weight[nh] = high_weight
                else:
                    sel_weight[nh] = timbre_weight

                if sel_weight[nh] > thresh_dB:
                    # Synthesis of harmonics is only applied if the selected weight is above the selected threshold
                    # for peak searching.
                    # Apply gain with difference from selected weight and magnitude of f0.
                    region_r_2 = region_r * (10 ** ((sel_weight[nh] - detected_f0_values[f0found[n] - 1, n]) / 20))
                    # Pad to compensate for the delay generated by the fractional delay filter.
                    region_r_pad = np.append(region_r_2, np.zeros(int(orderfracfilter / 2)))
                    # Filter magnitude region with Fractional Delay Filter using a Lagrange Interpolator.
                    region_r_filt = np.convolve(region_r_pad, h_lagrange, 'same')
                    # Keep interpolated values only.
                    region_r_filt = region_r_filt[int(orderfracfilter / 2):int(len(region_r_filt)) + 1]

                    start = int(shiftleft[nh])
                    end = int(shiftright[nh])

                    # Magnitude for synthesized frequency response
                    r_fsynth = np.abs(fsynth[start:end]) + (region_r_filt - np.abs(fsynth[start:end])) * freq_window
                    # Phase for synthesized frequency response
                    phi_fsynth = region_phi_x
                    # Calculate synthesized frequency response with Euler's form
                    fsynth[start:end] = r_fsynth * np.exp(1j * phi_fsynth)

                # ESTO LO AGREGO SOLO PARA QUE FUNCIONE LO DEL a MÁS ABAJO. CHEQUEAR.
                shiftright[nh] = right + np.floor(binshift) - 1

            # If harmonics were detected:
            elif detected_harmonics[nh, n] != 0:
                harmonic = detected_harmonics[nh, n]  # Detected harmonic
                harmonic_bin = round(harmonic)
                shiftleft[nh] = harmonic_bin - delta
                shiftright[nh] = harmonic_bin + delta
                start = int(shiftleft[nh])
                end = int(shiftright[nh]) + 1
                harmonicregion = f[start:end]
                harmonicregion_r = np.abs(harmonicregion)
                harmonicregion_phi = np.angle(harmonicregion)

                timbre_weight = target_weights_timbre[round(detected_harmonics[nh, n]), n]
                new_freq_bin[nh] = round(detected_harmonics[nh, n])

                low_weight = fixed_low[round(detected_harmonics[nh, n]), n]
                high_weight = fixed_high[round(detected_harmonics[nh, n]), n]

                # If weighting for harmonic is below ENV_low or above ENV_high,
                # adjust its value to keep it between these two envelopes.

                if timbre_weight < low_weight:
                    sel_weight[nh] = low_weight
                elif timbre_weight > high_weight:
                    sel_weight[nh] = high_weight
                else:
                    sel_weight[nh] = timbre_weight

                # Enhancement of harmonics is only applied if the detected harmonic values are below the selected
                # weight and if the selected weight is higher than the harmonic threshold detection level.
                if detected_harmonics_values[nh, n] < sel_weight[nh] and sel_weight[nh] > thresh_dB:
                    harmonicregion_r_gain = harmonicregion_r * 10 ** (
                            (sel_weight[nh] - detected_harmonics_values[nh, n]) / 20)
                    r_fsynth = harmonicregion_r + (harmonicregion_r_gain - harmonicregion_r) * freq_window
                    phi_fsynth = harmonicregion_phi
                    fsynth[start:end] = r_fsynth * np.exp(1j * phi_fsynth)

                accum_phases[start:end] = harmonicregion_phi

        a = []
        b = np.arange(len(f))
        for nh in np.arange(num_harmonics):
            a = np.append(a, np.arange(start=int(shiftleft[nh]), stop=int(shiftright[nh]) + 1)).astype('int')
        b = np.delete(b, a)  # Remove indices where there are harmonics?
        for i in b:
            accum_phases[i] = 0  # Where there aren't synthesized harmonics, accumulated phase is set to 0.

    YL[:, n] = fsynth

ys_proc = istft(YL, nHop, win, win) * nWin  # Multiply by nWin to compensate for line 357 ( f = Xs[:,n] / nWin )

# End of tonal processing (PV) #

# Tonal and transient reconstruction #
ys_proc = np.concatenate(
    (np.zeros(int(delay_transients)), ys_proc))  # Add delay caused by transient processing to tonal signal

# If separation method is Median Filtering, process noise signal; otherwise (MCA), ignore.
if sep_method == 'Median':
    yn = np.concatenate(
        (np.zeros(int(delay_transients)), yn))  # Add delay caused by transient processing to noise signal

    lengths = [len(ys_proc), len(yn), len(yt_proc)]
    min_length = np.min(lengths)  # Find minimum length among length of tonal, noise and transient signals

    # Make all signals the same length
    ys_proc = ys_proc[0:min_length]
    yt_proc = yt_proc[0:min_length]
    yn = yn[0:min_length]

    # Add processed tonal, transient and noise signals
    y_VBS = ys_proc + yt_proc + yn

elif sep_method == 'MCA':
    # Add delay to transient signal caused by STFT of tonal signal.
    # This is because in MCA no STFT is performed on the transient signal,
    # and therefore there is no nWin delay as in the case of Median Filter separation.
    # UPDATE (8/8/22) THIS IS NOT NECESSARY ANYMORE, SEE LINES 132-143.
    #yt_proc = np.concatenate((np.zeros(nWin), yt_proc))

    lengths = [len(ys_proc), len(yt_proc)]
    min_length = np.min(lengths)  # Find minimum length among length of tonal and transient signals

    # Make all signals the same length
    ys_proc = ys_proc[0:min_length]
    yt_proc = yt_proc[0:min_length]

    # Add processed tonal and transient signals
    y_VBS = ys_proc + yt_proc

# Resample to original sample rate
y_VBS = resample(y_VBS, Fs2, Fs)

# Apply delay to HPF input
delay = np.ceil(nWin * Fs / Fs2 + delay_transients * Fs / Fs2).astype('int')
x_hp = np.concatenate((np.zeros(delay), x_hp))

# NO LONGER NEEDED - THIS DELAY IS APPLIED A FEW LINES ABOVE TO THE yt_proc IF SEP METHOD IS MCA
# If separation method is MCA, apply a delay to the y_VBS signal
# This is because of the delay due to nWin which happens in the STFT and ISTFT stages
# of the median filter based separation process.
#if sep_method == 'MCA':
#    delay_MCA = np.ceil(nWin * Fs / Fs2).astype('int')
#    y_VBS = np.concatenate((np.zeros(delay_MCA), y_VBS))
#################################################################################################

# Construct output signal by adding VBS-processed and high-pass filtered inputs.
if len(x_hp) > len(y_VBS):
    y = x_hp[0:len(y_VBS)] + y_VBS
else:
    y = x_hp + y_VBS[0:len(x_hp)]

# Loudspeaker simulation filter
N_lspk = 3000  # Filter order for loudspeaker simulation high-pass and low-pass filters

# x_filt, b = HPFlspk(x, N=N_lspk, Fc=500, Fs=Fs)         # Original signal, high pass filtered (Fc = 500) para que sea más obvia la falta de graves
x_filt, b = HPFlspk(x, N=N_lspk, Fc=Fcc, Fs=Fs)         # Original signal, high pass filtered (Fc = Fcc)
y_filt, b = HPFlspk(y, N=N_lspk, Fc=Fcc, Fs=Fs)         # VBS-processed signal, high pass filtered
y_filt_low, b = LPFlspk(x, N=N_lspk, Fc=Fcc, Fs=Fs)     # Original signal, low pass filtered

# Calculate delay for final VBS processed signal, which consists of three separate delays:
# 1st, the delay caused by the tonal and transient processing: delay
# 2nd, the delay caused by the first low pass filter: N(LPF1) = N1
# 3rd, the delay caused by the loudspeaker simulation filter: N(HPFlspk) = N_lspk

delay_end = int(delay + N1 / 2 + N_lspk / 2)

#y_filt = y_filt[delay_end:-1]  # Old version, changed because it had 2 samples less than the Matlab version
y_filt = y_filt[delay_end-1:]   # Final VBS-enhanced signal, filtered by loudspeaker simulation filter

# Calculate delay for original signals (only considering the delay caused by the loudspeaker simulation filter)
delay_end_2 = int(N_lspk / 2)

x_filt = x_filt[delay_end_2:len(y_filt) + delay_end_2]  # High-passed version (Anchor for Bass Intensity test)

y_filt_low = y_filt_low[delay_end_2:len(y_filt) + delay_end_2]

y_darre = y_filt_low + y_filt  # Resulting signal with original low frequency components

if len(x) > len(y_darre):
    x_ref = x[0:len(y_darre)]  # Original signal with modified length
else:
    x_ref = x

############## Anchor 1 signal: add transient and tonal parts, adjust length and resample  ########################
# Ideas para la Anchor 1:
# - NLD FUNCTION que cause intermodulation distortion (IMD) - DONE
# - ENVELOPE MATCHING MALO: que produzca recorte de la señal - DONE
# - PV: ???
yt_dist_proc = yt_dist_proc[0:min_length]
y_VBS_dist = yt_dist_proc + ys_proc
y_VBS_dist = resample(y_VBS_dist, Fs2, Fs)
# Adding VBS-processed (distorted) and high-pass filtered inputs.
if len(x_hp) > len(y_VBS_dist):
    y_dist = x_hp[0:len(y_VBS_dist)] + y_VBS_dist
else:
    y_dist = x_hp + y_VBS_dist[0:len(x_hp)]
y_dist, b = HPFlspk(y_dist, N=N_lspk, Fc=Fcc, Fs=Fs)         # VBS-processed distorted signal, high pass filtered
y_dist = y_dist[delay_end-1:]  # Final VBS-enhanced distorted signal, filtered by loudspeaker simulation filter
###################################### END OF ANCHOR 1 SIGNAL PROCESSING ########################################

# END #

# Save output files #
now = datetime.datetime.now()
stamp = now.strftime("%d-%m-%Y_%H-%M-%S")

#audioWrite(f'audios/processed/{name}_original_{sep_method}_{stamp}.wav', x_ref, Fs)      # Reference signal
audioWrite(f'audios/processed/{name}_VB2_{sep_method}_{stamp}.wav', y_filt, Fs)          # VBS enhanced signal
#audioWrite(f'audios/processed/{name}_Anchor1_{sep_method}_{stamp}.wav', y_dist, Fs)      # Anchor 1 for Audio Quality test (highly distorted)
#audioWrite(f'audios/processed/{name}_Anchor2_{sep_method}_{stamp}.wav', x_filt, Fs)      # Anchor 2 for Bass Intensity test (high pass filtered)

# Stop timer and calculate elapsed time
toc = time.time()
elapsed = toc - tic
print(f'Process finished after {round(elapsed,2)} seconds.')