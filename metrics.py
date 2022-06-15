from sklearn.metrics import mean_squared_error
import numpy as np
from audioRead import audioRead
import musdb
import museval
import pandas as pd
from scipy.signal.windows import hann
from decomposeSTN import decomposeSTN
from plots import plot_audio
import matplotlib.pyplot as plt

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
xt_reference, fs, path, duration, frames, channels = audioRead('audios/museval/metronome_78BPM_less_beats.wav')
xs_reference, fs, path, duration, frames, channels = audioRead('audios/museval/pad.wav')
xt_fuzzy, fs, path, duration, frames, channels = audioRead('audios/museval/fuzzy/B_fuzzy_transient.wav')
xs_fuzzy, fs, path, duration, frames, channels = audioRead('audios/museval/fuzzy/B_fuzzy_tonal.wav')
xt_MCA, fs, path, duration, frames, channels = audioRead('audios/museval/MCA/B_MCA_transient.wav')
xs_MCA, fs, path, duration, frames, channels = audioRead('audios/museval/MCA/B_MCA_tonal.wav')
xt_MCA_fuzzy, fs, path, duration, frames, channels = audioRead('audios/museval/fuzzy_MCA/B_MCA_fuzzy_transient.wav')
xs_MCA_fuzzy, fs, path, duration, frames, channels = audioRead('audios/museval/fuzzy_MCA/B_MCA_fuzzy_tonal.wav')

# Zero padding to match shape of estimated sources and reference sources
s_pad = len(xs_reference)-len(xs_fuzzy)
t_pad = len(xt_reference)-len(xt_fuzzy)
xs_fuzzy = np.pad(xs_fuzzy, (0, s_pad), 'constant')
xt_fuzzy = np.pad(xt_fuzzy, (0, t_pad), 'constant')

# Truncate MCA audio files
xs_MCA = xs_MCA[:len(xs_reference)]
xt_MCA = xt_MCA[:len(xt_reference)]

xs_MCA_fuzzy = xs_MCA_fuzzy[:len(xs_reference)]
xt_MCA_fuzzy = xt_MCA_fuzzy[:len(xt_reference)]

references = [xs_reference, xt_reference, xs_reference, xt_reference, xs_reference, xt_reference]
estimates = [xs_fuzzy, xt_fuzzy, xs_MCA, xt_MCA, xs_MCA_fuzzy, xt_MCA_fuzzy]

nWin = len(xs_fuzzy)
nHop = len(xs_fuzzy)
SDR, ISR, SIR, SAR = museval.evaluate(references, estimates, win=nWin, hop=nHop)

#SDR, ISR, SIR, SAR = museval.evaluate(references, estimates)
#MSE = mean_squared_error(xilo_t_fuzzy, xilo_t_MCA)
print(MSE)
