import numpy as np
from audioRead import audioRead
from decomposeSTN import decomposeSTN
import scipy.signal as ss
import librosa as lib

x, Fs, path, duration, frames, channels = audioRead('audios/classical_mono_ref.wav')
nWin = 256
nHop = 32
nOL = nWin-nHop
#X, Xs, Xt, Xn, Rs, Rt, Rn = decomposeSTN(x, 1, nWin, nHop, Fs)
X1 = ss.stft(x, Fs, 'hann', nWin, nOL)
X2 = lib.stft(x, nWin, nHop, nWin,'hann')
print('done')