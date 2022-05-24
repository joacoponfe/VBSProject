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
from plots import plot_response, plot_spectrogram, plot_audio, plot_harmonics, plot_frame
from NLD import NLD


# Load input audio
# x, Fs, path, duration, frames, channels = audioRead('audios/music/classical_mono_ref.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/music/jazz_mono_ref.wav')
x, Fs, path, duration, frames, channels = audioRead('audios/music/pop_mono_ref.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/music/rock_mono_ref.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/test/pluck.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/test/xilo.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/test/piano_44kHz.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/xilo_MCA/xilo_transient_MCA_4096.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/museval/metronome_pad_mix.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/museval/MCA/tonal_museval.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/museval/MCA/transient_museval.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/NLD_tests/tone_1kHz_Amp_1.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/NLD_tests/tone_1kHz_Amp_0.8.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/NLD_tests/tone_1kHz_Amp_0.5.wav')
# x, Fs, path, duration, frames, channels = audioRead('audios/NLD_tests/tone_1kHz_Amp_0.3.wav')


# Stereo to mono conversion
x = x.transpose()
if len(x) == 2:
    x = 0.5 * (x[0] + x[1])

# Define parameters
num_harmonics = 5  # Number of processed harmonics
# Fcc = 150  # Cut-off frequency [Hz] (MOLINER)
Fcc = 250  # Cut-off frequency [Hz] (Según lo que escribí en el Marco Teórico)
GdB = 8  # NLD gain for transient calibration [dB]
thresh_dB = -70  # Magnitude threshold for peak search [dB]
inharmonicity_tolerance = 0.05  # Tolerance parameter (tau) for harmonic detection [%]
alpha_low = 7  # Lower weighting limit in dB/oct
alpha_high = 2  # Upper weighting limit in dB/oct
freq_window_Hz = 55  # Size of the frequency window for harmonic enhancement [Hz]
extra_harmonic_gain = 3  # Extra harmonic gain for tonal calibration

# Splitting and resampling

N1 = 2000  # Filter order for 1st HPF and LPF
Fc1 = 2000  # Cut-off frequency for 1st HPF and LPF

# High-pass filter
x_hp, b_hp1 = HPF1(x, N=N1, Fc=Fc1, Fs=Fs)

# Low-pass filter
x_lp, b_lp1 = LPF1(x, N=N1, Fc=Fc1, Fs=Fs)
#x_lp = x  # sólo para fuzzy tests

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

# Fuzzy separation (median filtering) based on Fitzgerald (2010).
# STFT parameters
#nWin = 2756  # Window size (2756 para mantener la relación de 16 entre 4096/256 y 44100/2756)
#nWin = 1024
nWin = 256
Ra = 32  # Hop size
noverlap = nWin - Ra  # Overlap size
S = 1  # Time-Scale Modification Factor (TSM)
win = hann(nWin)

X, Xs, Xt, Xn, Rs, Rt, Rn = decomposeSTN(x_lp, S, nWin, Ra, Fs2)

# ISTFT
yt = istft(Xt, Ra, win, win)
yn = istft(Xn, Ra, win, win)
ys = istft(Xs, Ra, win, win)
yfull = istft(X, Ra, win, win)

#########################################################################
# Para testear la separación
# Tomamos desde la muestra nº nWin en adelante, para que se alineen los audios.
#audioWrite('audios/museval/fuzzy_MCA/transient_MCA_Fuzzy.wav', yt[nWin:], Fs2)
#audioWrite('audios/museval/fuzzy_MCA/tonal_MCA_Fuzzy.wav', ys[nWin:], Fs2)
#audioWrite('audios/museval/fuzzy_MCA/noise_MCA_Fuzzy.wav', yn[nWin:], Fs2)
#audioWrite('audios/museval/fuzzy_MCA/full_MCA_Fuzzy.wav', yfull[nWin:], Fs2)
#########################################################################

# # PLOT #
#
# plot_audio(x,Fs,'Pluck')
# plt.show()
# tt = (len(x)/Fs)/np.shape(X)[1]
# xticks = np.asarray([0, 100, 200])
# xticklabels = np.round(xticks*tt, 2)
# xticklabels = [str(i) for i in xticklabels]
# plt.figure()
# plt.subplot(2, 2, 1)
# plt.title('Original $X(m,k)$')
# plt.imshow(abs(X), cmap='gray')
# #plt.xticks(xticks, xticklabels)
# plt.xlim([0, 200])
# plt.ylabel('k')
# plt.xlabel('m')
# plt.subplot(2, 2, 2)
# plt.title('Tonal $X_s(m,k)$')
# plt.imshow(abs(Xs), cmap='gray')
# #plt.xticks(xticks, xticklabels)
# plt.xlim([0, 200])
# #plt.ylabel('k')
# plt.xlabel('m')
# plt.subplot(2, 2, 3)
# plt.title('Transitorio $X_t(m,k)$')
# plt.imshow(abs(Xt), cmap='gray')
# #plt.xticks(xticks, xticklabels)
# plt.xlim([0, 200])
# plt.ylabel('k')
# plt.xlabel('m')
# plt.subplot(2, 2, 4)
# plt.title('Ruido $X_n(m,k)$')
# plt.imshow(abs(Xn), cmap='gray')
# #plt.xticks(xticks, xticklabels)
# plt.xlim([0, 200])
# #plt.ylabel('k')
# plt.xlabel('m')
# plt.show()

# Transient processing (NLD)
# LPF and HPF splitting
Nt = 500  # Filter order
SidelobeAtten = 40  # Window Side Lobe attenuation
wint = chebwin(Nt + 1, SidelobeAtten)  # Chebyshev Window
# Apply low pass and high pass filters
yt_lp, b_lp = LPFt(yt, Nt, SidelobeAtten, Fs=Fs2)
yt_hp, b_hp = HPFt(yt, Nt, SidelobeAtten, Fs=Fs2)

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
        yt_low_proc = h[n]  # For n = 0, yt_lp^0 = 1
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
# NBPF2 = 100
yt_low_proc_bpf, b_bp = BPFt(yt_low_proc_2, N=NBPF1, Fs=Fs2)

# PLOT RESPONSE OF BPF
# w_bp, h_bp = freqz(b_bp, [1.0])
# plot_response(Fs2, w_bp, h_bp, "BPF for Transient Content")
# plt.show()
# END PLOT RESPONSE #
# yt_low_proc_bpf = BPFt2(yt_low_proc_bpf, N=NBPF2, Fs=Fs2)

# Delay adjustment
# delay_low = np.round((NBPF1+NBPF2)/2)
delay_low = int(NBPF1 / 2)  # delay = Filter order / 2
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
    power_proc = np.sum(filtered_grain_low ** 2) / N_gain
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
np.where(difference_rs < thresh_dB, difference_rs, 0)
difference_rs = savgol_filter(difference_rs, int(N_gain / 2 + 1),
                              2)  # Savitzky-Golay filter para smoothing (ver si usar otro)
difference_rs = difference_rs[0:len(yt_low_proc_bpf)]

# Apply gain
G = 10 ** (difference_rs / 20)
yt_low_proc_gain = G * yt_low_proc_bpf

# Reconstruction
yt_proc = yt_hp + yt_low_proc_gain  # Adding high-pass filtered and processed low-pass signals together

delay_transients = Nt / 2 + delay_low  # Delay for transient signal: N(LPFt)/2 + N(BPF)/2

# End of transient processing with NLD #

# audioWrite('audios/classical_NLD_processed_28-04.wav', yt_proc, Fs2)

# Harmonic processing (PV) #

# Harmonic detection variables

# Initialization
nBins, nFrames = np.shape(X)

f0found = np.zeros(nFrames).astype('int')  # Number of f0 found, ranges from 0 to 3.
detected_f0 = np.zeros((3, nFrames))  # F0 frequency values [nBin], up to 3 possible values for each frame.
detected_f0_values = np.zeros((3, nFrames))  # F0 magnitude values [dB]
detected_harmonics = np.zeros((num_harmonics, nFrames))  # Harmonic frequency values [nBin]
detected_harmonics_values = np.zeros((num_harmonics, nFrames))  # Harmonic magnitude values [dB]
fixed_low = np.zeros((nBins, nFrames)) * (-1) * np.inf  # Weighting envelope low (?)
fixed_high = np.zeros((nBins, nFrames)) * np.inf  # Weighting envelope high (?)
accum_phases = np.zeros(nBins)  # Accumulated phases
YL = np.zeros((nBins, nFrames))  # Synthesized tonal spectrogram

# Estos son los índices/bins correspondientes a estas frecuencias, al parecer (?)
f0max = Fcc * nWin / Fs2
f0min = f0max / 4

# Harmonic enhancement variables
freq_window_size = int(np.round(freq_window_Hz * nWin / Fs2))
if freq_window_size % 2 == 0:  # If window is even-sized
    freq_window_size = freq_window_size + 1  # Becomes odd-sized
freq_window = tukey(freq_window_size, 0.75)  # w_ROI (ROI = region of influence) (CHEQUEAR)
for i in np.argwhere(freq_window == 0):
    freq_window[i] = 1e-3

# Harmonic weighting variables
target_weights_timbre = np.zeros((nBins, nFrames))
numBands = 7
# Range of frequencies where the weighting will be applied (in Hz)
# These specify the center frequencies (not edges) of the lowest and highest filters.
range = [Fcc / 4, Fcc * num_harmonics]

# Filter bank, Mel scale TODO revisar la FB, no coincide con Matlab. Ya está hecho la conversión a Bark y lo de center freqs.
FB, (cf, freqs) = compute_mat(num_bands=numBands, freq_min=range[0], freq_max=range[1], num_fft_bands=nWin,
                              sample_rate=Fs2, scale='bark')
# cf debe estar en Hz!!!
cf = bark_to_hertz(cf)  # Listo

# FB2 = mel(sr=Fs2, n_fft=nWin, n_mels=numBands, fmin=range[0], fmax=range[1]) # Función de librosa
# Falta arreglar FB y que cf sea correcto (para eso las freq_min y freq_max de entrada son CENTER freqs, no EDGES).


FBB = np.zeros((nBins, numBands))
FBB = FB.transpose()[:nBins, :]  # Mel Matrix for multiplying with Xs
# FBB = FBB/np.sum(FBB,1)
# cfbins = np.round(cf * nWin / Fs2) + 1
FBB = FBB / np.sum(FBB, 0)  # Una especie de normalización?
cfbins = np.round(cf * nWin / Fs2)  # Frequency bins corresponding to center frequencies of filterbank.

# Tonal processing
# Step 1: Fundamental and Harmonic detection
for n in np.arange(nFrames):
    n = 39  # DESPUES SACAR, ES PARA PRUEBAS
    f = Xs[:, n] / nWin  # Por qué divide por nWin
    r = np.abs(f)  # Magnitude spectrum

    bark = np.dot(FBB.transpose(), r)

    # All local maxima in the magnitude spectrum are selected as peaks
    # Fmin = Fcc / 4
    [exact, exact_peak] = peakdetect(r, Fs2, thresh_dB, Fcc / 4)

    # Possible improvement: change thresh_dB level depending on the dynamic range of the signal.

    # Create vector of frequency bins from first to last center frequency of bark filters (xq = query points)
    xq = np.arange(cfbins[0], cfbins[-1] + 1)

    # Interpolation of peak position (index) in Bark frequency scale
    interpbark = np.zeros(nBins)
    #fx = interp1d(cfbins, bark, kind='slinear')  # Interpolation function bark = f(cfbins) TODO see if kind can be changed to pchip (shape-preserving piecewise cubic interpolation)
    #interpbark_xq = fx(xq)  # Values of interpolation at query points (xp)
    interpbark_xq = pchip_interpolate(cfbins, bark, xq)
    interpbark[int(xq[0]):int(xq[-1] + 1)] = interpbark_xq  # Place interpolated values into interpbark vector (len = nBins)
    interpbark = 20 * np.log10(interpbark)  # dB

    # Get locations and magnitudes of detected peaks that are below f0max (candidates for F0).
    locations_fundamental_all = exact[np.where(exact < f0max)[0]]  # Evaluate all locations of detected peaks
    peak_fundamental_all = exact_peak[np.where(exact < f0max)[0]]  # Evaluate all magnitudes of detected peaks

    if np.size(peak_fundamental_all) != 0:
        # The maximum peak from the set of detected peaks is selected as the
        # fundamental candidate.
        maxim = np.max(peak_fundamental_all)  # Magnitude of maximum
        loc = np.argmax(peak_fundamental_all)  # Index of maximum
        maximloc = locations_fundamental_all[loc]  # Frequency bin (?) of maximum
        alimit = maximloc * inharmonicity_tolerance
        borders = np.array([1 / 3, 1 / 2, 1]) * f0max

        # Evaluates whether the candidate is the true f0 component,
        # or whether is could be the second harmonic of another peak in a lower frequency. (DONDE ESTÁ ESTO?)

        if maximloc / 3 > f0min:
            # ¿Qué hace esto? ¿Por qué dividido 3?
            f0cmin = (maximloc - alimit) / 3
            f0cmax = (maximloc + alimit) / 3
            a = np.min(abs(exact - maximloc / 3))
            s = np.argmin(abs(exact - maximloc / 3))
            if f0cmin < exact[s] < f0cmax:
                b = np.where(exact[s] <= borders)[0][0]  # Find first index of locations that are below borders
                detected_f0[b, n] = exact[s]
                detected_f0_values[b, n] = exact_peak[s]
                f0found[n] = b

        if f0found[n] == 0 and maximloc / 2 > f0min:
            f0cmin = (maximloc - alimit) / 2
            f0cmax = (maximloc + alimit) / 2
            a = np.min(abs(exact - maximloc / 2))
            s = np.argmin(abs(exact - maximloc / 2))
            if f0cmin < exact[s] < f0cmax:
                b = np.where(exact[s] <= borders)[0][0]  # Find first index of locations that are below borders
                detected_f0[b, n] = exact[s]
                detected_f0_values[b, n] = exact_peak[s]
                f0found[n] = b

        if f0found[n] == 0:
            b = np.where(maximloc <= borders)[0][0]
            detected_f0[b, n] = maximloc
            detected_f0_values[b, n] = maxim
            f0found[n] = b

        # Search all respective harmonics of f0
        # f_k = k * f0

        for h in np.arange(num_harmonics):
            # For the number of harmonics set, search for all respective harmonics of f0,
            # and, if these are within the region determined by the inharmonicity parameter,
            # save as detected harmonics.

            k = h + 5 - f0found[
                n]  # El 5 está porque f0found puede valer hasta 3, y en tal caso queremos que el primer armónic (no F0) sea la F0 multiplicada por 2
            pos = detected_f0[f0found[n], n] * k

            a = np.min(abs(exact - pos))  # Position of first harmonic
            b = np.argmin(abs(exact - pos))  # Index of first harmonic

            alimit = pos * inharmonicity_tolerance

            # If position of possible detected harmonic is within the region
            # of inharmonicity tolerance, save as detected harmonic,
            # otherwise, discard.

            if a < alimit:
                detected_harmonics[h, n] = exact[b]  # Position of harmonic
                detected_harmonics_values[h, n] = exact_peak[b]  # Value of harmonic
            else:
                detected_harmonics[h, n] = 0
                detected_harmonics_values[h, n] = 0

        # A partir de acá me falta estudiar qué está pasando
        interpbark = interpbark + (maxim + 10 ** (extra_harmonic_gain / 20)) - interpbark[int(np.round(maximloc + 1))]

    else:
        interpbark = interpbark * (-1) * np.inf

    start_value = interpbark[int(np.floor(f0max)) + 1]  # Chequear el +1
    fixed_low[0:int(np.floor(f0max)), n] = (-1) * np.inf
    fixed_high[0:int(np.floor(f0max)), n] = np.inf
    fixed_low[int(np.floor(f0max)) + 1:nBins, n] = start_value - alpha_low * np.arange(
        nBins - np.floor(f0max) - 1) / np.floor(f0max)
    fixed_high[int(np.floor(f0max)) + 1:nBins, n] = start_value - alpha_high * np.arange(
        nBins - np.floor(f0max) - 1) / np.floor(f0max)

    target_weights_timbre[:, n] = interpbark  # Assign weighting values.

    fsynth = f

    # Initialization
    shiftleft = np.zeros(num_harmonics)
    shiftright = np.zeros(num_harmonics)

    new_freq_bin = np.zeros(num_harmonics)
    sel_weight = np.zeros(num_harmonics)

    if f0found[n] != 0:
        fundamental_exact = detected_f0[f0found[n], n]  # Exact value of fundamental
        fundamental_bin = np.floor(fundamental_exact)  # Frequency bin where fundamental is located

        # Define limits of region of influence for the shifting.
        # So far it has been defined as a rectangular window of fixed size freq_window_size.
        # This could be a possible area of improvement:
        #   - Different freq_window_size for different frequencies?
        #   - Different type of window?

        delta = np.floor(freq_window_size / 2)
        left = int(fundamental_bin - delta)
        right = int(fundamental_bin + delta)

        region = f[left:right + 1]  # Region of influence (ROI)
        region_r = np.abs(region)  # Magnitude
        region_phi = np.angle(region)  # Phase

        for nh in np.arange(num_harmonics):
            beta = nh + 5 - f0found[n]  # Lo mismo que para definir el k
            if detected_harmonics[nh, n] == 0:  # No harmonics where detected, so they will be generated
                newfreq = beta * fundamental_exact  # Location of new harmonic
                binshift = newfreq - fundamental_exact  # How much shifting of the fundamental is necessary.
                # binshift refers to a number of samples, but it is most likely not an integer number.
                # This is the reason why we need to use a fractional delay filter, to interpolate magnitude
                # values for the instants between the discrete sampled values.

                shift = binshift * 2 * np.pi / nWin  # bin shift in Hz? rad/s?

                orderfracfilter = 4  # Order of the fractional delay filter
                # Falta el fractional delay filter (Lagrange).
                # The fractional delay filter is used for a more precise shifting of the frequency bins to the
                # exact target frequency, due to the rounding errors on shifting FFT bins.

                shiftleft[nh] = left + np.floor(binshift)  # Bin value for left extreme of the shifted region (?)
                shiftright[nh] = right + np.floor(binshift)  # Bin value for right extreme of the shifted region (?)

                # Phase unwrapping
                region_phi = np.unwrap(region_phi)  # Unwrap radian phase such that adjacent differences are never
                # greater than pi by adding 2kpi for some integer k.
                region_phi_pad = np.append(region_phi, np.zeros(int(orderfracfilter / 2)))  # Padding to compensate for
                # delay generated by fractional delay filter.
                # region_phi_filtered = filter(secondOrderFrac, region_phi_pad)
                region_phi_filtered = region_phi_pad  # Temporary!
                region_phi_filtered = region_phi_filtered[int(orderfracfilter / 2):len(region_phi_filtered + 1)]
                p0 = accum_phases[round(newfreq)]  # Get accumulated phases for harmonic location
                pu = p0 + Ra * shift  # Phase of shifted partial?
                region_phi_x = region_phi_filtered + pu  # Accumulated phase is added to phase of filtered region (?)
                accum_phases[int(shiftleft[nh]):int(shiftright[nh]) + 1] = np.ones(len(region)) * pu

                timbre_weight = target_weights_timbre[
                    round(newfreq) + 1, n]  # Get harmonic weighting values (chequear el +1)
                new_freq_bin[nh] = round(newfreq) + 1  # Chequear el + 1
                sel_weight[nh] = timbre_weight

                low_weight = fixed_low[round(newfreq) + 1, n]
                high_weight = fixed_high[round(newfreq) + 1, n]

                # If selected weight is not within low and high thresholds, adjust accordingly.
                if timbre_weight < low_weight:
                    sel_weight[nh] = low_weight
                elif timbre_weight > high_weight:
                    sel_weight[nh] = high_weight

                if sel_weight[
                    nh] > thresh_dB:  # In this case, the weighting is above the selected threshold for peak searching.
                    # Apparently, synthesis of harmonics is only applied if this condition is True.
                    region_r_2 = region_r * (10 ** ((sel_weight[nh] - detected_f0_values[
                        f0found[n], n]) / 20))  # Apply gain with difference from selected weight and value of f0
                    region_r_pad = np.append(region_r_2, np.zeros(int(orderfracfilter / 2)))
                    # region_r_filt = filter(secondOrderFrac, region_r_pad)
                    region_r_filt = region_r_pad  # Temporary!
                    region_r_filt = region_r_filt[int(orderfracfilter / 2):int(len(region_r_filt)) + 1]

                    start = int(shiftleft[nh])
                    end = int(shiftright[nh]) + 1
                    r_fsynth = np.abs(fsynth[start:end]) + (region_r_filt - np.abs(
                        fsynth[start:end])) * freq_window  # Magnitude for synthesized frequency response
                    phi_fsynth = region_phi_x  # Phase for synthesized frequency response
                    fsynth[start:end] = r_fsynth * np.exp(1j * phi_fsynth)

            # If harmonics were detected:
            elif detected_harmonics[nh, n] != 0:
                harmonic = detected_harmonics[nh, h]  # Detected harmonic
                harmonic_bin = round(harmonic) + 1  # Chequear el +1
                shiftleft[nh] = harmonic_bin - delta
                shiftright[nh] = harmonic_bin + delta
                start = int(shiftleft[nh])
                end = int(shiftright[nh]) + 1
                harmonicregion = f[start:end]
                harmonicregion_r = np.abs(harmonicregion)
                harmonicregion_phi = np.angle(harmonicregion)

                timbre_weight = target_weights_timbre[round(detected_harmonics[nh, h] + 1), n]  # Chequear el +1
                new_freq_bin[nh] = round(detected_harmonics[nh, n]) + 1
                sel_weight[nh] = timbre_weight

                low_weight = fixed_low[round(detected_harmonics[nh, n]) + 1, n]
                high_weight = fixed_high[round(detected_harmonics[nh, n] + 1), n]

                if timbre_weight < low_weight:
                    sel_weight[nh] = low_weight
                elif timbre_weight > high_weight:
                    sel_weight[nh] = high_weight

                # Revisar lo que sigue:
                # Synthesis of harmonics is only applied if the detected harmonic values are below the selected
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

yL = istft(YL, Ra, win, win) * nWin  # ¿Por qué multiplica por nWin?

# End of tonal processing (PV) #

# Tonal and transient reconstruction #

yL = np.concatenate((np.zeros(int(delay_transients)), yL))  # Add delay caused by transient processing to tonal signal
yn = np.concatenate((np.zeros(int(delay_transients)), yn))  # Add delay caused by transient processing to noise signal
lengths = [len(yL), len(yn), len(yt_proc)]
min_length = np.min(lengths)  # Minimum length of tonal, noise and transient signals

# Make all signals the same length
yL = yL[0:min_length]
yn = yn[0:min_length]
yt_proc = yt_proc[0:min_length]

# Add processed tonal, transient and noise signals
y_VBS = yL + yn + yt_proc

# Resample to original sample rate
y_VBS = resample(y_VBS, Fs2, Fs)

# Apply delay to HPF input
delay = np.ceil(nWin * Fs / Fs2 + delay_transients * Fs / Fs2).astype('int')
x_hp = np.concatenate((np.zeros(delay), x_hp))

# Construct output signal by adding VBS-processed and high-pass filtered inputs.
if len(x_hp) > len(y_VBS):
    y = x_hp[0:len(y_VBS)] + y_VBS
else:
    y = x_hp + y_VBS[0:len(x_hp)]

# Loudspeaker simulation filter
N_lspk = 3000  # Filter order for loudspeaker simulation high-pass and low-pass filters

x_filt, b = HPFlspk(x, N=N_lspk, Fc=Fcc, Fs=Fs)
y_filt, b = HPFlspk(y, N=N_lspk, Fc=Fcc, Fs=Fs)
y_filt_low, b = LPFlspk(y, N=N_lspk, Fc=Fcc, Fs=Fs)

delay_end = int(delay + N1 / 2 + N_lspk / 2)  # N(LPF1) = N1 = 2000; N(HPFlspk) = N_lspk = 3000
y_filt = y_filt[delay_end:-1]  # Final signal, filtered by loudspeaker simulation filter

delay_end_2 = int(N_lspk / 2)

x_filt = x_filt[delay_end_2:len(y_filt) + delay_end_2]  # High-passed version

y_filt_low = y_filt_low[delay_end_2:len(y_filt) + delay_end_2]

y_darre = y_filt_low + y_filt  # Resulting signal with low components

if len(x) > len(y_darre):
    x2 = x[0:len(y_darre)]  # Original signal with modified length

# END #

# Save output file #
audioWrite('audios/piano_VBS_05_05_22_2.wav', y_VBS, Fs)
