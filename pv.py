import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, stft, istft, chebwin, firwin
from librosa import resample
import soundfile as sf
import dtcwt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Peak detection
def peak_detect(signal, threshold=150):
    """Function that finds peaks (for instance in a spectrogram).
    A peak corresponds to a sample that is greater than its two nearest neighbors."""
    k = 2
    indices = []
    while k < len(signal) - 2:
        seg = np.asarray(signal[k-2:k+3])
        if np.max(seg) < threshold:
            k += 2
        else:
            if seg.argmax() == 2:  # Maximum is located in the center of the segment
                indices.append(k)
                k += 2
        k += 1
    return indices


def lfBoost(x, Fs, Fc, GdB, pv1, pv2, pv3):
    delay = 0
    xTot = x.transpose()
    FsTot = Fs
    # Copy and downsample signal for more efficient processing
    Fs = 4096
    x = resample(xTot, FsTot, Fs)

    # Define target cutoff frequency
    FcV = Fc

    # Generate the hanning window at appropriate length
    N = int(2**np.ceil(np.log2(abs(Fs/4))))  # Window length
    HA = int(N/8)                            # Hop size
    w = np.hanning(N)

    # Apply LPF to the input
    N1 = 200
    SidelobeAtten = 60
    #win = chebwin(N1+1, SidelobeAtten)
    b = firwin(N1, Fc/(Fs/2), pass_zero='lowpass', window=('chebwin', SidelobeAtten), scale='True')
    x = lfilter(b, [1.0], x)

    # Add silence to beginning and end of input
    L = len(x)
    x = np.concatenate((np.zeros(N), x, np.zeros(N - L % HA)))/max(abs(x))

    # Initializations
    yL = np.zeros(L)      # Output vector

    # Determine number of windows needed
    pend = len(x) - N            # Analysis end index

    # Set pitch shift parameter range (increments of 1, 1 = fund)
    alphaL = pv1
    alphaH = pv2
    step = pv3

    # Set up frequency vector
    fVec = np.linspace(0, Fs, N)

    # Define the frequencies for four target/non-target ranges
    Fc1 = 20
    Fc2 = FcV   # Cut off frequency
    nR = 4      # Number of ranges
    st = (Fc2-Fc1)/nR
    rL = Fc1 + st*np.linspace(0, nR-1, nR)     # Start freqs of each range
    rH = Fc1 + st*np.linspace(1, nR, nR)       # Stop freqs of each range

    # Process each target range separately and sum the outputs
    for z in range(nR):
        # Determine target/non-target frequency bins
        indVec = np.arange(np.where(rL[z] <= fVec)[0][0], np.where(rH[0] <= fVec)[0][0]+1, 1)
        otherVec = []
        for n in range(len(fVec)):
            if np.where(n == indVec)[0].size == 0:  # Check whether this array is empty
                otherVec.append(n)

        # Initialize temporary output vector
        yT = np.zeros(len(x))

        #Run the procedure to generate the required harmonics
        for alpha in np.arange(alphaL, alphaH, step):
            L1 = np.floor(N/2)                                       # Nyquist evaluation length
            shift = alpha
            while rH[z]*shift < FcV:
                shift = shift + 1
            oVec = [otherVec[i] for i in np.where(otherVec <= L1)[0]]
            oVec = oVec[0:-1]  # Elimina la ultima muestra para que coincida con la long. del vector de Matlab
            omega = 2*np.pi*HA*np.arange(0, L1, 1)/N                    # Sinusoid reconstruction vector
            pin = 0
            pout = 0
            phi0 = np.zeros(int(L1))   # Initial measured phase
            r0 = np.zeros(int(L1))     # Initial magnitude
            psi = np.zeros(int(L1))    # Initial phase
            grain = np.zeros(N)   # Analysis window vector
            res = np.zeros(HA)    # Resulting signal vector

            # Run the procedure
            while pin < pend:
                grain = x[pin:pin+N]*w                      # Analysis window
                fc1 = np.fft.fft(np.fft.fftshift(grain))    # FFT of analysis window (w/circular shift)
                f = fc1[0:int(L1)]                          # Analysis window up to Nyquist
                r = np.abs(f)                               # FFT magnitude
                phi = np.angle(f)                           # FFT phase
                delta_phi = omega + ((phi-phi0-omega)+np.pi) % (-2*np.pi) + np.pi      #Unwrapped phase diff.
                delta_r = (r-r0)/HA                         # Change in magnitude
                delta_psi = delta_phi/HA                    # Phase increment
                # Apply linear interpolation over hop size
                for k in range(HA):
                    # Shift each frequency bin
                    r0 = r0 + delta_r
                    for i in oVec:
                        psi[i] = psi[i] + delta_psi[i]
                    for j in indVec:
                        psi[j] = psi[j] + shift*delta_psi[j]
                    res[k] = np.dot(r0, np.sin(psi))
                phi0 = phi     # Save previous meas. phase
                r0 = r         # Save previous magnitude
                psi = (psi+np.pi) % (-2*np.pi) + np.pi      # Unwrapped phase increment
                # Create time domain output with appropriate gain
                yT[pout:pout+HA] = yT[pout:pout+HA] + res
                pin = pin + HA
                pout = pout + HA

        # Normalize output and cut out silence at beginning and end
        yT = np.real(yT[int(N/2+HA):int(N/2+HA+L)])

        # Add to total output
        yL = yL + yT

    # Normalize y
    yL = yL/max(abs(yL))

    # Create and apply inverse A-Weighting filter

    return x


x, Fs = sf.read('Audios/rock_x_stereo.wav')
x = x.transpose()
# Ensure file is mono
if np.shape(x)[0] == 2:
    x = 0.5*(x[0]+x[1])
y = lfBoost(x, Fs, 200, 6, 1, 2, 3)
print('done')
#def phasevocoder(x,nHopA, nHopS, Rt, transient_threshold,max_decay_lenth,ola_coef):

# Read input file (stereo)
x, fs = sf.read('Audios/jazz_x_stereo.wav')
#xL = [i[0] for i in x]  # Left channel
#xR = [i[1] for i in x]  # Right channel

# Define parameters
fc = 250  # Cut-off frequency
alpha_low = 7  # Lower weighting limit in dB/oct
alpha_high = 2  # Upper weighting limit in dB/oct
freq_window_hz = 55  # Size of frequency window for harmonic enhancement
extra_harmonic_gain = 3  # Extra harmonic gain for tonal calibration

# Apply LPF
x_filt = butter_lowpass_filter(xL, fc, fs, 5)

# Downsampling
fs2 = 4096  # Target sampling rate
xL_resampled = resample(x_filt, fs, fs2)


# Define window size (nWin), overlap factor (OLF), pitch ratio, synthesis hop size (nHopS) and analysis hop size (nHopA)
pitch_ratio = 4
OLF = 4
nWin = 2048  # Window size (samples)
nHopS = nWin/OLF  # Synthesis hop size
nHopA = int(nHopS/pitch_ratio)  # Analysis hop size

# Create window
win = np.hanning(nWin)

# Signal blocks for synthesis hopsizeing and output (initialization)
#delta_phase = np.zeros(nWin/2+1)                # delta phase
#syn_phase = np.zeros(nWin/2+1, dtype=complex)   # synthesis phase angle

#k = np.linspace(0, nWin/2, nWin/2+1)            # ramp
#last_phase = np.zeros(nWin/2+1)                 # last frame phase
#accum_phase = np.zeros(nWin/2+1)                # accumulated phase
#current_frame = np.zeros(nWin/2+1)
#expected_phase = k*2*np.pi*nHopA/nWin           # Expected phase

# Apply the STFT to the signal
f, t, Zxx = stft(xL_resampled, fs, 'hann', nWin, nHopS)
# Get vectors of magnitude and phase
X_mag = np.abs(Zxx)
X_phase = np.angle(Zxx)

# DT-CWT TEST ######
# Input array X's size must be a multiple of 2
X = xL  # Grabs the left channel only

# Yo voy a hacer que el tamaño del input array X sea POTENCIA de 2, para simplificar los cálculos que siguen
N = len(X)
if np.log2(N).is_integer() == False:
    power = int(np.ceil(np.log2(N)))  # Rounds up to the nearest integer
    N2 = 2**power  # Define new length
    X = np.append(X, np.zeros(N2-N))  # Zero pad
# Perform Wavelet transform up to log2(N) levels

lvls = int(np.ceil(np.log2(N)))
transform = dtcwt.Transform1d()
vecs_t = transform.forward(X, nlevels=lvls, include_scale='True')
#vecs_t = transform.forward(X, nlevels=5, include_scale='True')

# Get the highpass (detail) and lowpass (approximation) coefficients of each level
DT1C_HP = vecs_t.highpasses # Complex highpass coefficients
DT1C_LP = vecs_t.scales  # Real lowpass coefficients (why not complex? :/ )

# Since the amount of samples in each step decreases, we need to make
# sure that we repeat the samples 2^i times where i is the level so
# that at each level, we have the same amount of transformed samples
# as in the first level. This is only necessary because of plotting.

coeffs = DT1C_LP  # Select coefficients

#cc = np.abs(np.array(coeffs[-1]))  # Last level coefficients (real)

A = np.zeros((lvls, 2**lvls))  # Initialize empty array
for i in range(lvls-1):
    power = pow(2, lvls-1-i)
    selected_row = np.abs(np.array(coeffs[lvls-1-i]))
    row_full = np.tile(selected_row, [power, 1])
    A[i] = row_full.transpose()

plt.figure()
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('Scalogram')
t = np.linspace(start=0, stop=N2/fs, num=N2)  # Time vector
f = np.logspace(start=lvls-1, stop=0, num=lvls, base=2)
X, Y = np.meshgrid(t, f)
plt.pcolormesh(X, Y, A, shading='auto')
plt.tight_layout()
plt.show()

# Since the amount of samples in each step decreases, we need to make
# sure that we repeat the samples 2^i times where i is the level so
# that at each level, we have the same amount of transformed samples
# as in the first level. This is only necessary because of plotting.


# Calculate phase and magnitude
#DT1C_mag = np.abs(DT1C)
#DT1C_phase = np.angle(DT1C)
# END OF DTCWT TEST ######

# Analyse first and second window and extract peaks
X0_mag = [i[0] for i in X_mag]
X0_phase = [i[0] for i in X_phase]

X1_mag = [i[1] for i in X_mag]
X1_phase = [i[1] for i in X_phase]

y0 = np.asarray(X0_mag)
rms0 = np.sqrt(np.mean(y0**2))
y1 = np.asarray(X1_mag)
rms1 = np.sqrt(np.mean(y1**2))

thold0 = rms0  # Define threshold
thold1 = rms1
indices0 = peak_detect(X0_mag, thold0)  # Find indices of peaks
f0_peaks = [f[i] for i in indices0]  # Find frequencies of peaks
X0_peaks = [X0_mag[i] for i in indices0]  # Find magnitudes of peaks

indices1 = peak_detect(X1_mag, thold1)
f1_peaks = [f[i] for i in indices1]
X1_peaks = [X1_mag[i] for i in indices1]

# Plot results
freq = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
freq_labels = ['31.5',' 63', '125', '250', '500', '1k', '2k', '4k', '8k', '16k']
fig, axs = plt.subplots(3, 1)
axs[0].plot(f, X0_mag)
axs[0].set_xscale('log')
axs[0].set_xticks(freq)
axs[0].set_xticklabels(freq_labels)
axs[0].scatter(f0_peaks, X0_peaks, color='r')
axs[0].hlines(thold0, 0, 30000, color='r', linestyles='dashed')
axs[0].set_xlim([0, 20000])

axs[1].plot(f, X1_mag)
axs[1].set_xscale('log')
axs[1].set_xticks(freq)
axs[1].set_xticklabels(freq_labels)
axs[1].scatter(f1_peaks, X1_peaks, color='r')
axs[1].hlines(thold1, 0, 30000, color='r', linestyles='dashed')
axs[1].set_xlim([0, 20000])

#  ACA IRIA EL PLOT DE LA DTCWT PARA COMPARAR
axs[2].plot(DT1C_mag)

fig.tight_layout()
plt.show()

# Pitch shifting
common_peaks = np.intersect1d(f0_peaks, f1_peaks)  # Find common peaks between two consecutive windows
print('f0_peaks: ', f0_peaks)
print('f1_peaks: ', f1_peaks)
print('common peaks: ', common_peaks)

peak_indices = [np.where(f == i)[0][0] for i in common_peaks]  # Find indices of common peaks
print('peak_indices: ', peak_indices)
print('first peak freq.: ', f[peak_indices[0]])

theta_0 = X0_phase[peak_indices[0]]
theta_1 = X1_phase[peak_indices[0]]
delta_theta = theta_1 - theta_0

delta_t = t[1] - t[0]

fn = []
for i in np.arange(100):  # Falta automatizar la forma de determinar el rango
    fx = (delta_theta + 2*np.pi*i)/(2*np.pi*delta_t)
    fn.append(fx)

print('fn array: ', fn)

# Find closest element to value of peak in fn array
value = f[peak_indices[0]]
absolute_val_array = np.abs(fn-value)
smallest_difference_idx = absolute_val_array.argmin()
closest_fn = fn[smallest_difference_idx]
print('Estimate freq.: ', closest_fn)

# Harmonic search
fc = 150  # Cutoff frequency (Hz)
K = 6  # Number of harmonics to be searched
tau = 0.05  # Tolerance parameter (relaxes the search in the cases where the signal is not exactly harmonic.)

f0 = closest_fn

fk = [k*f0 for k in np.arange(K)[1:]]  # Array of harmonics to be searched
print('fk:', fk)

for i in fk:
    x = np.argmin(np.abs(common_peaks-i))  # Searches for possible harmonic among found peaks
    x = common_peaks[x]
    print(x)
    if np.abs(x-i) < tau*i:
        print(x+' has been detected as a harmonic.')

# NO USAR COMMON PEAKS, USAR OTRA COSA (f0_peaks, f1_peaks?)

# Define Region of Influence (ROI) window


