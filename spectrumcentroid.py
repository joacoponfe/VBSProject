from librosa import feature, load, times_like
import matplotlib.pyplot as plt

#y, sr = load('audios/classical_ref.wav')
y, sr = load('audios/hiphop_ref.wav')
#y, sr = load('audios/rock_ref.wav')
#y, sr = load('audios/jazz_ref.wav')
centroid = feature.spectral_centroid(y=y, sr=sr)
t = times_like(centroid)

plt.figure()
plt.plot(t, centroid.T)
plt.show()

