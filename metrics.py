from sklearn.metrics import mean_squared_error
import numpy as np
from audioRead import audioRead
import musdb
import museval
import pandas as pd
import soundfile as sf
from scipy.signal.windows import hann
from decomposeSTN import decomposeSTN
from plots import plot_audio
import matplotlib.pyplot as plt
import matlab.engine


def spectral_centroid(x, samplerate=44100):
    magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1]) # positive frequencies
    return np.sum(magnitudes*freqs) / np.sum(magnitudes) # return weighted mean


# Initialize Matlab Engine and add folders with .m scripts to directory
eng = matlab.engine.start_matlab()
PEAQ = eng.genpath('PQevalAudio-v1r0')
PEASS = eng.genpath('peass-master-v2.0.1')
eng.addpath(PEAQ, nargout=0)
eng.addpath(PEASS, nargout=0)

# Determine reference and test files (fs must be 48 kHz)
#ref = 'PQevalAudio-v1r0/PQevalAudio/audios_48k/AimeeNorwich_drums_lp_4096.wav'
#test = 'PQevalAudio-v1r0/PQevalAudio/audios_48k/AimeeNorwich_mix_lp_4096_MCA_transient_bior6.8.wav'

# Temporal Envelope Matching Tests
# (reference is the target signal before NLD processing; test is the signal after NLD + Envelope Matching)
ref = 'audios/TEM_tests/audios_48k/rock_Median_Transient_LP_4096_Target.wav'
test = 'audios/TEM_tests/audios_48k/rock_Median_Transient_LP_4096_Target.wav'

# Calculate PEAQ metrics
PEAQ_res = eng.PQevalAudio(ref, test)

# Calculate PEASS metrics
PEASS_res = eng.PEASS_ObjectiveMeasure(ref, test)
PEASS_res.pop('decompositionFilenames')

# Calculate Audio Spectral Centroid (ASC)
y, fs = sf.read(test)
centroid = spectral_centroid(y, fs)


# Make global dictionary
res_dict = {"Reference": ref, "Test": test}
res_dict.update(PEAQ_res)   # Add results from PEAQ dictionary
res_dict.update(PEASS_res)  # Add results from PEASS dictionary

# Make pandas dataframe
#res_df = pd.DataFrame.from_dict(res_dict, orient='columns')
res_df = pd.DataFrame([res_dict])

# Append result to .csv file
# append data frame to CSV file
res_df.to_csv('metrics.csv', mode='a', index=True, header=False)

# print message
print("Data appended successfully.")

# STIMULUS A
# mix_A, fs, path, duration, frames, channels = audioRead('audios/museval/stimulusA.wav')
# xt_reference_A, fs, path, duration, frames, channels = audioRead('audios/museval/metronome_78BPM.wav')
# xs_reference_A, fs, path, duration, frames, channels = audioRead('audios/museval/pad.wav')
#xt_fuzzy, fs, path, duration, frames, channels = audioRead('audios/museval/fuzzy/A_fuzzy_transient.wav')
#xs_fuzzy, fs, path, duration, frames, channels = audioRead('audios/museval/fuzzy/A_fuzzy_tonal.wav')
#xt_MCA, fs, path, duration, frames, channels = audioRead('audios/museval/MCA/A_MCA_transient.wav')
#xs_MCA, fs, path, duration, frames, channels = audioRead('audios/museval/MCA/A_MCA_tonal.wav')
#xt_MCA_fuzzy, fs, path, duration, frames, channels = audioRead('audios/museval/fuzzy_MCA/A_MCA_fuzzy_transient.wav')
#xs_MCA_fuzzy, fs, path, duration, frames, channels = audioRead('audios/museval/fuzzy_MCA/A_MCA_fuzzy_tonal.wav')

# STIMULUS B
# mix_B, fs, path, duration, frames, channels = audioRead('audios/museval/stimulusB.wav')
#xt_reference, fs, path, duration, frames, channels = audioRead('audios/museval/metronome_78BPM_less_beats.wav')
#xs_reference, fs, path, duration, frames, channels = audioRead('audios/museval/pad.wav')
#xt_fuzzy, fs, path, duration, frames, channels = audioRead('audios/museval/fuzzy/B_fuzzy_transient.wav')
#xs_fuzzy, fs, path, duration, frames, channels = audioRead('audios/museval/fuzzy/B_fuzzy_tonal.wav')
#xt_MCA, fs, path, duration, frames, channels = audioRead('audios/museval/MCA/B_MCA_transient.wav')
#xs_MCA, fs, path, duration, frames, channels = audioRead('audios/museval/MCA/B_MCA_tonal.wav')
#xt_MCA_fuzzy, fs, path, duration, frames, channels = audioRead('audios/museval/fuzzy_MCA/B_MCA_fuzzy_transient.wav')
#xs_MCA_fuzzy, fs, path, duration, frames, channels = audioRead('audios/museval/fuzzy_MCA/B_MCA_fuzzy_tonal.wav')

# Zero padding to match shape of estimated sources and reference sources
#s_pad = len(xs_reference)-len(xs_fuzzy)
#t_pad = len(xt_reference)-len(xt_fuzzy)
#xs_fuzzy = np.pad(xs_fuzzy, (0, s_pad), 'constant')
#xt_fuzzy = np.pad(xt_fuzzy, (0, t_pad), 'constant')

# Truncate MCA audio files
#xs_MCA = xs_MCA[:len(xs_reference)]
#xt_MCA = xt_MCA[:len(xt_reference)]

#xs_MCA_fuzzy = xs_MCA_fuzzy[:len(xs_reference)]
#xt_MCA_fuzzy = xt_MCA_fuzzy[:len(xt_reference)]

#references = [xs_reference, xt_reference, xs_reference, xt_reference, xs_reference, xt_reference]
#estimates = [xs_fuzzy, xt_fuzzy, xs_MCA, xt_MCA, xs_MCA_fuzzy, xt_MCA_fuzzy]

#nWin = len(xs_fuzzy)
#nHop = len(xs_fuzzy)
#SDR, ISR, SIR, SAR = museval.evaluate(references, estimates, win=nWin, hop=nHop)

#SDR, ISR, SIR, SAR = museval.evaluate(references, estimates)
#MSE = mean_squared_error(xilo_t_fuzzy, xilo_t_MCA)
#print(MSE)
