#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 11:53:51 2022

@author: guillem
"""

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

ref, fs = sf.read('audios/y_reference.wav')
targ, fs = sf.read('audios/y_target.wav')

ref_env = np.abs(hilbert(ref))
targ_env = np.abs(hilbert(targ))

g_envs = ref_env/targ_env

targ_g = g_envs*targ

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(7, 5))
ax[0].plot(ref)
ax[0].plot(ref_env, color='k')
ax[1].plot(targ)
ax[1].plot(targ_env, color='k')
ax[2].plot(g_envs)
ax[3].plot(targ_g)
ax[3].plot(ref_env, color='k')

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(targ_g[10000:11000])

plt.show()
sf.write('audios/target_g.wav', targ_g, fs)

