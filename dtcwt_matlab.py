import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn, lfilter, firwin
import soundfile


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
    xc, wc = dualTree1D(x, J, Faf, af)
    OUTPUT
        xc = approximation/lowpass coefficients
        wc = detail/highpass coefficients
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

    wc = [W0, W1]
    xc = [X0, X1]
    return xc, wc


def idualtree(wc, J, Fsf, sf):
    """Inverse Dual-tree Complex DWT.
    USAGE:
        y = idualtree(w, J, Fsf, sf)
    INPUT:
        wc - DWT coefficients
        J - number of stages
        Fsf - synthesis filters for the 1st stage
        sf - synthesis filters for remaining stages
    OUTPUT:
        y - output signal
    """

    # Tree 1 (i = 0)
    y0 = wc[0][J]
    for j in reversed(range(1, J)):
        y0 = sfb(y0, wc[0][j], sf[0])
    y0 = sfb(y0, wc[0][0], Fsf[0])

    # Tree 2 (i = 1)
    y1 = wc[1][J]
    for j in reversed(range(1, J)):
        y1 = sfb(y1, wc[1][j], sf[1])
    y1 = sfb(y1, wc[1][0], Fsf[1])

    # Normalization
    y = (y0+y1)/np.sqrt(2)
    return y


def instFreq(wc, xc, J):
    """Calculates instantaneous frequencies across each scale of the DT-CWT.
    Based on Evans et al. - Time-Scale Modification of Audio Signals Using the DT-CWT.
    INPUTS:
        wc : complex DT-CWT highpass coefficients
        xc : complex DT-CWT lowpass coefficients
        J : number of scales/levels (integer)
    OUTPUTS:
        Y : list of instantaneous frequencies across each scale
    """
    Y = []
    omega3 = []
    for j in range(J):
        xi_j = 100  # Averiguar el valor real, varía con cada nivel
        A = wc[j]
        B = xc[j]
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
            C = xc[j - 1][0:int(len(xc[j - 1]) / 2)]  # Tomo la mitad del vector para que no haya quilombos de shapes
            psiC = np.angle(C)
            delta_psi = psiC - psiB - xi_j
            psi = np.unwrap(delta_psi)
            omega2 = xi_j + psi
            omega3 = np.concatenate((omega2, omega3))
        omega_full = np.concatenate((omega, omega3))
        Y.append(omega_full)
        return Y


def scalogram(xc, wc, t_n, J, N):
    """Make a scalogram plot for the DT-CWT result.
    INPUTS:
        - xc = DT-CWT approximation coefficients (lowpass)
        - wc = DT-CWT detail coefficients (highpass)
        - t_n = Duration (in seconds) of the input signal
        - J = number of levels of the DT-CWT transform
        - N = input signal length"""
    W = []
    X = []
    for i in range(J):
        w_cplx = wc[0][i] + 1j * wc[1][i]
        x_cplx = xc[0][i] + 1j * xc[1][i]  # Armar el vector de coeficientes complejo tomando la parte real e imaginaria de c/ nivel
        W.append(w_cplx)
        X.append(x_cplx)

    # Armar matriz de coeficientes
    coeffs = []
    coeffs.append(X[-1])
    for i in range(J):
        coeffs.append(W[-(i + 1)])

    cc = np.array([coeffs[-1]])
    for i in range(J - 1):
        cc = np.concatenate([cc, np.array([np.repeat(coeffs[J - 1 - i], pow(2, i + 1))])])
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
    y_plot = np.logspace(start=J - 1, stop=0, num=J, base=2)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
    plt.pcolormesh(X_plot, Y_plot, cc_real)
    plt.colorbar()

    use_log_scale = False  # Cambiar a False si quiero escala lineal

    if use_log_scale:
        plt.yscale('log')
    else:
        yticks = [pow(2, i) for i in range(J)]
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
    return None


def centerFreq(w):
    """Computes the center frequency of a given wavelet signal.
    INPUT
        w: wavelet signal
    OUTPUT
        fc: center frequency"""
    w_fft = np.abs(np.fft.fft(w)) ** 2
    w_freq = np.linspace(0, 1, len(w_fft))
    idxmax = np.argmax(w_fft)
    fc = w_freq[idxmax]
    return fc

Faf, Fsf = FSfarras()
af, sf = dualfilt1()

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

scales = np.arange(1, J+1)
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






