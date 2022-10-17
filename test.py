import numpy as np
import matplotlib.pyplot as plt
from plots import plot_audio
from utils import audioRead

Fs = 4096

ys_Median_jazz = audioRead('audios/jazz_mono_ref_Median_tonal.wav')[0]
yt_Median_jazz = audioRead('audios/jazz_mono_ref_Median_transient.wav')[0]
ys_Median_rock = audioRead('audios/rock_mono_ref_Median_tonal.wav')[0]
yt_Median_rock = audioRead('audios/rock_mono_ref_Median_transient.wav')[0]
ys_Median_classic = audioRead('audios/classical_mono_ref_Median_tonal.wav')[0]
yt_Median_classic = audioRead('audios/classical_mono_ref_Median_transient.wav')[0]
ys_Median_pop = audioRead('audios/pop_mono_ref_Median_tonal.wav')[0]
yt_Median_pop = audioRead('audios/pop_mono_ref_Median_transient.wav')[0]

ys_MCA_jazz = audioRead('audios/MCA/jazz_mono_ref_lp_4096_MCA_tonal_bior6.8-2.wav')[0]
yt_MCA_jazz = audioRead('audios/MCA/jazz_mono_ref_lp_4096_MCA_transient_bior6.8-2.wav')[0]
ys_MCA_rock = audioRead('audios/MCA/rock_mono_ref_lp_4096_MCA_tonal_bior6.8-2.wav')[0]
yt_MCA_rock = audioRead('audios/MCA/rock_mono_ref_lp_4096_MCA_transient_bior6.8-2.wav')[0]
ys_MCA_classic = audioRead('audios/MCA/classical_mono_ref_lp_4096_MCA_tonal_bior6.8-2.wav')[0]
yt_MCA_classic = audioRead('audios/MCA/classical_mono_ref_lp_4096_MCA_transient_bior6.8-2.wav')[0]
ys_MCA_pop = audioRead('audios/MCA/pop_mono_ref_lp_4096_MCA_tonal_bior6.8-2.wav')[0]
yt_MCA_pop = audioRead('audios/MCA/pop_mono_ref_lp_4096_MCA_transient_bior6.8-2.wav')[0]

plt.figure(figsize=(9.5,3.8))
plt.figtext(0.5, 0.95, 'Filtro de mediana', ha='center', va='center', fontsize=12)
plt.figtext(0.5, 0.5, 'MCA', ha='center', va='center', fontsize=12)
plt.subplot(2,2,1)
plot_audio(ys_Median_pop,Fs,'Tonal')
#plt.xlim([3,6])
plt.ylim([-.5,.5])
plt.xlabel('')
plt.subplot(2,2,2)
plot_audio(yt_Median_pop,Fs,'Transitorio')
#plt.xlim([3,6])
plt.ylim([-1,1])
plt.xlabel('')
plt.subplot(2,2,3)
plot_audio(ys_MCA_pop,Fs)
#plt.xlim([3,6])
plt.ylim([-.5,.5])
plt.subplot(2,2,4)
plot_audio(yt_MCA_pop,Fs)
#plt.xlim([3,6])
plt.ylim([-1,1])
plt.tight_layout()
plt.subplots_adjust(hspace=0.6)
plt.savefig('plots/Median_MCA_pop.svg')
plt.savefig('plots/Median_MCA_pop.png')
plt.show()


