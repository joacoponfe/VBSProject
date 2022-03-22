import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn, lfilter, firwin
import soundfile
import pywt


def afb(x, af):
    """Analysis filter bank
        N-point vector (N even);
        The resolution should be 2x filter length

        af -- analysis filter
        af_lp lowpass filter (even length)
        af_hp highpass filter (even length)

        lo: Low frequency
        hi: High frequency

        upfirdn: upsample, apply FIR filter and downsample
        scipy.signal.upfirdn(h,x,up,down)
            h: 1-D FIR filter coefficients
            x: input signal array
            up: upsampling rate
            down: downsampling rate
        """
    N = int(len(x))
    L = int(len(af)/2)
    x = np.roll(x, -L)

    # Lowpass filter
    af_lp = [i[0] for i in af]
    lo = upfirdn(af_lp, x, 1, 2)
    lo[:L] = lo[int(N/2):int(N/2)+L] + lo[:L]
    lo = lo[:int(N/2)]

    # Highpass filter
    af_hp = [i[1] for i in af]
    hi = upfirdn(af_hp, x, 1, 2)
    hi[:L] = hi[int(N/2):int(N/2)+L] + hi[:L]
    hi = hi[:int(N/2)]
    return lo, hi


def sfb(lo, hi, sf):
    """Synthesis filter bank

    lo - low frequency signal
    hi - high frequency signal
    sf - synthesis filters
    sf_lp - lowpass filter (even length)
    sf_hp - highpass filter (even length)

    y - output
    """

    N = int(2*len(lo))
    L = int(len(sf))

    sf_lp = [i[0] for i in sf]
    sf_hp = [i[1] for i in sf]
    lo = upfirdn(sf_lp, lo, 2, 1)
    hi = upfirdn(sf_hp, hi, 2, 1)
    y = lo + hi
    y[:L-2] = y[:L-2] + y[N:N+L-2]
    y = y[:N]
    y = np.roll(y, int(1-L/2))
    return y


def FSfarras():
    """Filters used in the first stage of the DT-CWT."""
    af0 = np.array([[0, 0], [-0.08838834764832, -0.01122679215254],
            [0.08838834764832, 0.01122679215254],
            [0.69587998903400, 0.08838834764832],
            [0.69587998903400, 0.08838834764832],
            [0.08838834764832, -0.69587998903400],
            [-0.08838834764832, 0.69587998903400],
            [0.01122679215254, - 0.08838834764832],
            [0.01122679215254, -0.08838834764832],
            [0, 0]])
    #sf0 = np.flip(af0, 0)
    sf0 = af0

    af1 = np.array([[0.01122679215254, 0],
            [0.01122679215254, 0],
            [-0.08838834764832, -0.08838834764832],
            [0.08838834764832, -0.08838834764832],
            [0.69587998903400, 0.69587998903400],
            [0.69587998903400, -0.69587998903400],
            [0.08838834764832, 0.08838834764832],
            [-0.08838834764832, 0.08838834764832],
            [0, 0.01122679215254],
            [0, -0.01122679215254]])
    #sf1 = np.flip(af1, 0)
    sf1 = af1

    af = [af0, af1]
    sf = [sf0, sf1]
    return af, sf


def dualfilt1():
    """Analysis and synthesis filters for the remaining stages of the DT-CWT.
    As published in [KIN-1999].
    af[i]: analysis filters for tree i
    sf[i]: synthesis filters for tree i
    Note: af[1] is the reverse of af[0]."""
    af0 = np.array([[0.03516384000000, 0],
            [0, 0],
            [-0.08832942000000, -0.11430184000000],
            [0.23389032000000, 0],
            [0.76027237000000, 0.58751830000000],
            [0.58751830000000, -0.76027237000000],
            [0, 0.23389032000000],
            [-0.11430184000000, 0.08832942000000],
            [0, 0],
            [0, -0.03516384000000]])
    sf0 = np.flip(af0, 0)

    af1 = np.flip(af0, 0)
    sf1 = np.flip(af1, 0)

    af = [af0, af1]
    sf = [sf0, sf1]
    return af, sf


def dualtree1D(x, J, Faf, af):
    """Dual-tree Complex Discrete Wavelet Transform.
    w = dualTree1D(x, J, Faf, af)
    OUTPUT
        w[j]: scale j (j = 0,...,J-1)
        w[j][i] tree i (i = 0, 1)
    INPUT
        x: 1-D signal
        J: number of stages
        Faf: first stage filter
        af: filter for remaining stages
        """
    # Normalization
    x = x/np.sqrt(2)

    # Initialization of lists
    W0 = []  # List for 1st tree (real) coefficients for each level
    W1 = []  # List for 2nd tree (imaginary) coefficients for each level
    X0 = []
    X1 = []

    # Tree 1 (i = 0)
    x0, w0 = afb(x, Faf[0])  # Applying first stage filter
    X0.append(x0)
    W0.append(w0)
    for k in range(1, J):
        x0, w0 = afb(x0, af[0])
        W0.append(w0)
        X0.append(x0)

    # w[J][0] = x0  # No entiendo por qué hace esto, es como que agrega un nivel más. Lo voy a ignorar por ahora
    W0.append(x0)

    # Tree 2 (i = 1)
    x1, w1 = afb(x, Faf[1])
    W1.append(w1)
    X1.append(x1)
    for k in range(1, J):
        x1, w1 = afb(x1, af[1])
        W1.append(w1)
        X1.append(x1)
    # w[J][1] = x1  # Ídem anterior
    W1.append(x1)

    w = [W0, W1]
    x = [X0, X1]
    return x, w


def idualtree(w, J, Fsf, sf):
    """Inverse Dual-tree Complex DWT.
    USAGE:
        y = idualtree(w, J, Fsf, sf)
    INPUT:
        w - DWT coefficients
        J - number of stages
        Fsf - synthesis filters for the 1st stage
        sf - synthesis filters for remaining stages
    OUTPUT:
        y - output signal
    """

    # Tree 1 (i = 0)
    y0 = w[0][J]
    for j in reversed(range(1, J)):
        y0 = sfb(y0, w[0][j], sf[0])
    y0 = sfb(y0, w[0][0], Fsf[0])

    # Tree 2 (i = 1)
    y1 = w[1][J]
    for j in reversed(range(1, J)):
        y1 = sfb(y1, w[1][j], sf[1])
    y1 = sfb(y1, w[1][0], Fsf[1])

    # Normalization
    y = (y0+y1)/np.sqrt(2)
    return y


def instFreq(W, X, J):
    '''Calculates instantaneous frequencies across each scale of the DT-CWT.
    Based on Evans et al. - Time-Scale Modification of Audio Signals Using the DT-CWT.
    INPUTS:
        W : complex DT-CWT highpass coefficients
        X : complex DT-CWT lowpass coefficients
        J : number of scales/levels (integer)
    OUTPUTS:
        Y : list of instantaneous frequencies across each scale
    '''
    Y = []
    omega3 = []
    for j in range(lvls):
        xi_j = 100  # Averiguar el valor real, varía con cada nivel
        A = W[j]
        B = X[j]
        # Calculate phases of each analysis window in the corresponding level
        psiA = np.angle(A)
        psiB = np.angle(B)
        # Calculate phase increments
        delta_psi = psiB - psiA - xi_j
        # Principal determination (between +/- π)
        psi = np.unwrap(delta_psi)
        # Instantaneous frequency
        omega = xi_j + psi
        if j > 0:  # Para niveles mayores al primero, se calcula un salto más, y se concatena con los anteriores.
            C = X[j - 1][0:int(len(X[j - 1]) / 2)]  # Tomo la mitad del vector para que no haya quilombos de shapes
            psiC = np.angle(C)
            delta_psi = psiC - psiB - xi_j
            psi = np.unwrap(delta_psi)
            omega2 = xi_j + psi
            omega3 = np.concatenate((omega2, omega3))
        omega_full = np.concatenate((omega, omega3))
        Y.append(omega_full)
        return Y


def scalogram(x, w, t_n, lvls):
    """Make a scalogram plot for the DT-CWT result.
    INPUTS:
        - x = DT-CWT approximation coefficients (lowpass)
        - w = DT-CWT detail coefficients (highpass)
        - t_n = Duration (in seconds) of the input signal
        - lvls = number of levels of the DT-CWT transform"""
    W = []
    X = []
    for i in range(lvls):
        w_cplx = w[0][i] + 1j * w[1][i]
        x_cplx = x[0][i] + 1j * x[1][i]  # Armar el vector de coeficientes complejo tomando la parte real e imaginaria de c/ nivel
        W.append(w_cplx)
        X.append(x_cplx)

    # Armar matriz de coeficientes
    coeffs = []
    coeffs.append(X[-1])
    for i in range(lvls):
        coeffs.append(W[-(i + 1)])

    cc = np.array([coeffs[-1]])
    for i in range(lvls - 1):
        cc = np.concatenate([cc, np.array([np.repeat(coeffs[lvls - 1 - i], pow(2, i + 1))])])
    cc_real = np.abs(cc)
    cc_imag = np.angle(cc)

    # Plot scalogram
    plt.figure()
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('DT-CWT')
    # X-axis has a linear scale (time)
    x_plot = np.linspace(start=0, stop=t_n, num=N // 2)
    # Y-axis has a logarithmic scale (frequency)
    y_plot = np.logspace(start=lvls - 1, stop=0, num=lvls, base=2)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
    plt.pcolormesh(X_plot, Y_plot, cc_real)
    plt.colorbar()

    use_log_scale = False  # Cambiar a False si quiero escala lineal

    if use_log_scale:
        plt.yscale('log')
    else:
        yticks = [pow(2, i) for i in range(lvls)]
        plt.yticks(yticks)

    plt.tight_layout()
    plt.show()
    return None


def filters_plot(Faf, af):
    # Create the frequency vectors of the filters (from 0 to 1)
    freq_Faf0 = np.linspace(0, 1, len(Faf[0]))
    freq_Faf1 = np.linspace(0, 1, len(Faf[1]))
    freq_af0 = np.linspace(0, 1, len(af[0]))
    freq_af1 = np.linspace(0, 1, len(af[1]))

    # Create the Fourier energy spectra of the filters
    fft_Faf0 = np.abs(np.fft.fft(Faf[0])) ** 2
    fft_Faf1 = np.abs(np.fft.fft(Faf[1])) ** 2
    fft_af0 = np.abs(np.fft.fft(af[0])) ** 2
    fft_af1 = np.abs(np.fft.fft(af[1])) ** 2

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('First stage analysis filters')
    labels = ['LP', 'HP']

    axs[0, 0].plot(Faf[0], label=labels)
    axs[0, 0].set_title('Faf0')
    axs[0, 0].legend()

    axs[0, 1].plot(freq_Faf0, fft_Faf0, label=labels)
    axs[0, 1].set_title('FFT of Faf0')
    axs[0, 1].legend()

    axs[1, 0].plot(Faf[1], label=labels)
    axs[1, 0].set_title('Faf1')
    axs[1, 0].legend()

    axs[1, 1].plot(freq_Faf1, fft_Faf1, label=labels)
    axs[1, 1].set_title('FFT of Faf1')
    axs[1, 1].legend()

    fig2, axs2 = plt.subplots(2, 2)
    fig2.suptitle('Remaining stages analysis filters')

    axs2[0, 0].plot(af[0], label=labels)
    axs2[0, 0].set_title('af0')
    axs2[0, 0].legend()

    axs2[0, 1].plot(freq_af0, fft_af0, label=labels)
    axs2[0, 1].set_title('FFT of af0')
    axs2[0, 1].legend()

    axs2[1, 0].plot(af[1], label=labels)
    axs2[1, 0].set_title('af1')
    axs2[1, 0].legend()

    axs2[1, 1].plot(freq_af1, fft_af1, label=labels)
    axs2[1, 1].set_title('FFT of af1')
    axs2[1, 1].legend()
    plt.show()

# Input signal (X) considerations
# Signal length must be a power of two
x, Fs = soundfile.read('Audios/jazz_x_stereo.wav')
x = x.transpose()
# If x is stereo, convert to mono
if len(x) == 2:
    x = 0.5*(x[0]+x[1])

N = len(x)  # Calculate length of signal
if not np.log2(N).is_integer():
    J = int(np.ceil(np.log2(N)))  # Rounds up to the nearest integer
    N2 = 2**J  # Define new length as nearest power of two
    x = np.append(x, np.zeros(N2-N))  # Zero pad

# Check that lowest frequency band is < 10 Hz.
lowest_freq = Fs/(2**J)
while lowest_freq > 10:
    # More levels/scales are required
    J = J + 1  # Increase level
    lowest_freq = Fs/(2**J)  # Recompute lowest frequency
    N2 = 2**J
    x = np.append(x, np.zeros(N2-N))


######## EJEMPLO DE STACK OVERFLOW #############
N = 128     # Number of samples
t_n = 2     # Signal duration [s]
# Create signal
xc = np.linspace(0, t_n, num=N)
xd = np.linspace(0, t_n, num=32)
sig = np.sin(2*np.pi * 64 * xc[:32]) * (1 - xd)
composite_signal3 = np.concatenate([np.zeros(32), sig[:32], np.zeros(N-32-32)])
lvls = int(np.ceil(np.log2(N)))
Faf, Fsf = FSfarras()       # 1st stage analysis and synthesis filters
af, sf = dualfilt1()        # Remaining stages anal. and synth. filters
x, w = dualtree1D(composite_signal3, lvls, Faf, af)


###### PLOT THE ANALYSIS FILTERS ########
# Create the frequency vectors of the filters (from 0 to 1)
freq_Faf0 = np.linspace(0, 1, len(Faf[0]))
freq_Faf1 = np.linspace(0, 1, len(Faf[1]))
freq_af0 = np.linspace(0, 1, len(af[0]))
freq_af1 = np.linspace(0, 1, len(af[1]))

# Create the Fourier energy spectra of the filters
fft_Faf0 = np.abs(np.fft.fft(Faf[0]))**2
fft_Faf1 = np.abs(np.fft.fft(Faf[1]))**2
fft_af0 = np.abs(np.fft.fft(af[0]))**2
fft_af1 = np.abs(np.fft.fft(af[1]))**2

fig, axs = plt.subplots(2, 2)
fig.suptitle('First stage analysis filters')
labels = ['LP', 'HP']

axs[0, 0].plot(Faf[0], label=labels)
axs[0, 0].set_title('Faf0')
axs[0, 0].legend()

axs[0, 1].plot(freq_Faf0, fft_Faf0, label=labels)
axs[0, 1].set_title('FFT of Faf0')
axs[0, 1].legend()

axs[1, 0].plot(Faf[1], label=labels)
axs[1, 0].set_title('Faf1')
axs[1, 0].legend()

axs[1, 1].plot(freq_Faf1, fft_Faf1, label=labels)
axs[1, 1].set_title('FFT of Faf1')
axs[1, 1].legend()


fig2, axs2 = plt.subplots(2, 2)
fig2.suptitle('Remaining stages analysis filters')

axs2[0, 0].plot(af[0], label=labels)
axs2[0, 0].set_title('af0')
axs2[0, 0].legend()

axs2[0, 1].plot(freq_af0, fft_af0, label=labels)
axs2[0, 1].set_title('FFT of af0')
axs2[0, 1].legend()

axs2[1, 0].plot(af[1], label=labels)
axs2[1, 0].set_title('af1')
axs2[1, 0].legend()

axs2[1, 1].plot(freq_af1, fft_af1, label=labels)
axs2[1, 1].set_title('FFT of af1')
axs2[1, 1].legend()

plt.show()

# Find values of fc (center frequencies) of each wavelet
FFTs = [fft_Faf0, fft_Faf1, fft_af0, fft_af1]
FREQs = [freq_Faf0, freq_Faf1, freq_af0, freq_af1]
FCs = []
for i in range(len(FFTs)):
    idxmax1 = np.argmax(FFTs[i][:, 0])
    idxmax2 = np.argmax(FFTs[i][:, 1])
    freq1 = FREQs[i][idxmax1]
    freq2 = FREQs[i][idxmax2]
    fcs = [freq1, freq2]
    FCs.append(fcs)

fcs_Faf0 = FCs[0]
fcs_Faf1 = FCs[1]
fcs_af0 = FCs[2]
fcs_af1 = FCs[3]

# Pseudo-frequencies initialization
PF_all = []

scales = np.arange(1, lvls+1)
for i in range(len(FCs)):
    PF = []
    for fc in FCs[i]:
        pf = [Fs*fc/a for a in scales]  # pseudo-frequencies
        PF.append(pf)
    PF_all.append(PF)

PF_Faf0 = PF_all[0]
PF_Faf1 = PF_all[1]
PF_af0 = PF_all[2]
PF_af1 = PF_all[3]

################################ SCALOGRAM PLOT ##################################
W = []
X = []
for i in range(lvls):
    w_cplx = w[0][i] + 1j*w[1][i]
    x_cplx = x[0][i] + 1j*x[1][i]   # Armar el vector de coeficientes complejo tomando la parte real e imaginaria de c/ nivel
    W.append(w_cplx)
    X.append(x_cplx)

# Armar matriz de coeficientes para graficar el scalogram
coeffs = []
coeffs.append(X[-1])
for i in range(lvls):
    coeffs.append(W[-(i+1)])

#cc = np.abs(np.array([coeffs[-1]]))
cc = np.array([coeffs[-1]])
for i in range(lvls - 1):
    #cc = np.concatenate(np.abs([cc, np.array([np.repeat(coeffs[lvls - 1 - i], pow(2, i + 1))])]))
    cc = np.concatenate([cc, np.array([np.repeat(coeffs[lvls - 1 - i], pow(2, i + 1))])])
cc_real = np.abs(cc)
cc_imag = np.angle(cc)

# Plot scalogram
plt.figure()
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('DT-CWT')
# X-axis has a linear scale (time)
x_plot = np.linspace(start=0, stop=t_n, num=N//2)
# Y-axis has a logarithmic scale (frequency)
y_plot = np.logspace(start=lvls-1, stop=0, num=lvls, base=2)
X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
plt.pcolormesh(X_plot, Y_plot, cc_real)
plt.colorbar()

use_log_scale = False        # Cambiar a False si quiero escala lineal

if use_log_scale:
    plt.yscale('log')
else:
    yticks = [pow(2, i) for i in range(lvls)]
    plt.yticks(yticks)

plt.tight_layout()
plt.show()

############################################################################
# Using the DT-CWT in the Phase-Vocoder Algorithm (from Evans et al.)
#Calculating the instantaneous frequencies across each scale
instFreq = []
omega3 = []
for j in range(lvls):
    xi_j = PF_af0[0][j]*2*np.pi  # Pseudo-frequency of level j converted to rad/s
    # Convert pseudofrequencies to
    A = W[j]
    B = X[j]
    # Calculate phases of each analysis window in the corresponding level
    psiA = np.angle(A)
    psiB = np.angle(B)
    # Calculate phase increments
    delta_psi = psiB - psiA - xi_j
    # Principal determination (between +/- π)
    psi = np.unwrap(delta_psi)
    # Instantaneous frequency
    omega = xi_j + psi
    if j > 0:  # Para niveles mayores al primero, se calcula un salto más, y se concatena con los anteriores.
        C = X[j-1][0:int(len(X[j-1])/2)]  # Tomo la mitad del vector para que no haya quilombos de shapes
        psiC = np.angle(C)
        delta_psi = psiC - psiB - xi_j
        psi = np.unwrap(delta_psi)
        omega2 = xi_j + psi
        omega3 = np.concatenate((omega2, omega3))
    omega_full = np.concatenate((omega, omega3))
    instFreq.append(omega_full)
print('k')


# PyWavelet comparison
# Custom wavelet
Faf_LP_tree0 = Faf[0][:, 0]
Faf_HP_tree0 = Faf[0][:, 1]
Faf_LP_tree1 = Faf[1][:, 0]
Faf_HP_tree1 = Faf[1][:, 1]

Fsf_LP_tree0 = Fsf[0][:, 0]
Fsf_HP_tree0 = Fsf[0][:, 1]
Fsf_LP_tree1 = Fsf[1][:, 0]
Fsf_HP_tree1 = Fsf[1][:, 1]

af_LP_tree0 = af[0][:, 0]
af_HP_tree0 = af[0][:, 1]
af_LP_tree1 = af[1][:, 0]
af_HP_tree1 = af[1][:, 1]

sf_LP_tree0 = sf[0][:, 0]
sf_HP_tree0 = sf[0][:, 1]
sf_LP_tree1 = sf[1][:, 0]
sf_HP_tree1 = sf[1][:, 1]

filterbank_FS_tree0 = [Faf_LP_tree0, Faf_HP_tree0, Fsf_LP_tree0, Fsf_HP_tree0]
filterbank_FS_tree1 = [Faf_LP_tree1, Faf_HP_tree1, Fsf_LP_tree1, Fsf_HP_tree1]
filterbank_tree0 = [af_LP_tree0, af_HP_tree0, sf_LP_tree0, sf_HP_tree0]
filterbank_tree1 = [af_LP_tree1, af_HP_tree1, sf_LP_tree1, sf_HP_tree1]

DT0_FS_wavelet = pywt.Wavelet(name='DT0_FS', filter_bank=filterbank_FS_tree0)
DT1_FS_wavelet = pywt.Wavelet(name='DT1_FS', filter_bank=filterbank_FS_tree1)
DT0_wavelet = pywt.Wavelet(name='DT0', filter_bank=filterbank_tree0)
DT1_wavelet = pywt.Wavelet(name='DT1', filter_bank=filterbank_tree1)


fc1 = pywt.central_frequency(DT0_FS_wavelet)
fc2 = pywt.central_frequency(DT1_FS_wavelet)
fc3 = pywt.central_frequency(DT0_wavelet)
fc4 = pywt.central_frequency(DT0_wavelet)

scalfreq = []
for i in range(1, lvls+1):
 freq = pywt.scale2frequency(DT0_FS_wavelet, i)
 scalfreq.append(freq*Fs)

print('m')
