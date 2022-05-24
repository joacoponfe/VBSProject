"""
Source: https://github.com/SiggiGue/pyfilterbank/blob/master/pyfilterbank/melbank.py
This module implements a Mel Filter Bank.
In other words it is a filter bank with triangular shaped bands
arnged on the mel frequency scale.

An example is shown in the following figure:

.. plot::

    from pylab import plt
    import melbank

    f1, f2 = 1000, 8000
    melmat, (melfreq, fftfreq) = melbank.compute_melmat(6, f1, f2, num_fft_bands=4097)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(fftfreq, melmat.T)
    ax.grid(True)
    ax.set_ylabel('Weight')
    ax.set_xlabel('Frequency / Hz')
    ax.set_xlim((f1, f2))
    ax2 = ax.twiny()
    ax2.xaxis.set_ticks_position('top')
    ax2.set_xlim((f1, f2))
    ax2.xaxis.set_ticks(melbank.mel_to_hertz(melfreq))
    ax2.xaxis.set_ticklabels(['{:.0f}'.format(mf) for mf in melfreq])
    ax2.set_xlabel('Frequency / mel')
    plt.tight_layout()

    fig, ax = plt.subplots()
    ax.matshow(melmat)
    plt.axis('equal')
    plt.axis('tight')
    plt.title('Mel Matrix')
    plt.tight_layout()


Functions
---------
"""

from numpy import abs, append, arange, insert, linspace, log10, round, zeros, where, array, asarray, dot, sqrt, sign, floor


def hertz_to_mel(freq):
    """Returns mel-frequency from linear frequency input.

    Parameter
    ---------
    freq : scalar or ndarray
        Frequency value or array in Hz.

    Returns
    -------
    mel : scalar or ndarray
        Mel-frequency value or ndarray in Mel

    """
    return 2595.0 * log10(1 + (freq / 700.0))


def mel_to_hertz(mel):
    """Returns frequency from mel-frequency input.

    Parameter
    ---------
    mel : scalar or ndarray
        Mel-frequency value or ndarray in Mel

    Returns
    -------
    freq : scalar or ndarray
        Frequency value or array in Hz.

    """
    return 700.0 * (10 ** (mel / 2595.0)) - 700.0


def bark_to_hertz(b):
    """Convert BARK frequency scale to Hertz.
    frq = bark_to_hertz(bark) converts a vector of frequencies in BARK scale to
    the corresponding values in Hz.
    Inputs:
        b: matrix of frequencies in BARK
    Outputs:
        f : Hz values
    Adapted from v_bark2freq.m (MATLAB function)
    by Mike Brookes © 2006-2010.
    """

    A = 26.81
    B = 1960
    C = -0.53
    E = A + C
    D = A * B
    P = (0.53 / (3.53 ** 2))
    V = 3 - 0.5 / P
    W = V ** 2 - 9
    Q = 0.25
    R = 20.4
    xy = 2
    S = 0.5 * Q / xy
    T = R + 0.5 * xy
    U = T - xy
    X = T * (1 + Q) - Q * R
    Y = U - 0.5 / S
    Z = Y ** 2 - U ** 2

    a = abs(b)

    # Low frequency correction
    m1 = where(a < 3)[0]
    for i in m1:
        a[i] = V + sqrt(W + a[i] / P)

    # High frequency correction
    m2 = where(a > X)[0]
    m1 = where(a > U)[0]
    for i in m2:
        a[i] = (a[i] + Q * R) / (1 + Q)
    for j in m1:
        a[j] = Y + sqrt(Z + a[j] / S)

    f = array(D * (E - a) ** (-1) - B)

    # Force to be odd
    f = f * sign(b)

    return f


def hertz_to_bark(f):
    """Convert Hertz to BARK frequency scale.
        bark = hertz_to_bark(frq) converts a vector of frequencies (in Hz) to
        the corresponding values in the BARK scale.
        Inputs:
            f : matrix of frequencies in Hz
        Outputs:
            b : bark values
        Adapted from v_frq2bark.m (MATLAB function)
        by Mike Brookes © 2006-2010.
        """

    A = 26.81
    B = 1960
    C = -0.53
    D = A * B
    P = (0.53 / 3.53 ** 2)
    Q = 0.25
    R = 20.4
    xy = 2
    S = 0.5 * Q / xy
    T = R + 0.5 * xy
    U = T - xy

    f = [f]  # Para que sea indexable
    g = abs(f)
    b = A * g / (B + g) + C

    # Low frequency correction
    m1 = where(b < 3)[0]
    for i in m1:
        b[i] = b[i] + P * (3 - b[i]) ** 2

    # High frequency correction
    m2 = where(b > T)[0]
    m1 = where(b > U)[0]
    for i in m2:
        b[i] = b[i] + S * (b[i] - U) ** 2
    for j in m1:
        b[j] = (1 + Q) * b[j] - Q * R

    # Force to be odd
    b = b * sign(f)

    return b


def barkfrequencies_bark_filterbank(num_bands, freq_min, freq_max, num_fft_bands, center=True):
    """Returns center frequencies and band edges for a bark filter bank.
    Parameters
    ----------
    num_bands : int
        Number of bark bands.
    freq_min : scalar
        Minimum frequency for the first band (Hz).
    freq_max : scalar
        Maximum frequency for the last band (Hz).
    num_fft_bands : int
        Number of fft bands.
    center : bool
        (Default: True). If True, freq_min and freq_max indicate center frequencies of lowest and highest
        bands (not edges).

    Returns
    -------
    center_frequencies_bark : ndarray
    lower_edges_bark : ndarray
    upper_edges_bark : ndarray

    """
    if center:
        bark_min_center = hertz_to_bark(freq_min)
        bark_max_center = hertz_to_bark(freq_max)
        delta_bark = abs(bark_max_center - bark_min_center) / (num_bands - 1.0)
        frequencies_bark = append(bark_min_center - delta_bark, bark_min_center + delta_bark * arange(0, num_bands + 1))
        center_frequencies_bark = frequencies_bark[1:-1]
        lower_edges_bark = frequencies_bark[:-2]
        upper_edges_bark = frequencies_bark[2:]
    else:
        bark_max = hertz_to_bark(freq_max)
        bark_min = hertz_to_bark(freq_min)
        delta_bark = abs(bark_max - bark_min) / (num_bands + 1.0)
        frequencies_bark = bark_min + delta_bark * arange(0, num_bands + 2)
        lower_edges_bark = frequencies_bark[:-2]
        upper_edges_bark = frequencies_bark[2:]
        center_frequencies_bark = frequencies_bark[1:-1]

    return center_frequencies_bark, lower_edges_bark, upper_edges_bark


def melfrequencies_mel_filterbank(num_bands, freq_min, freq_max, num_fft_bands, center=True):
    """Returns center frequencies and band edges for a mel filter bank
    Parameters
    ----------
    num_bands : int
        Number of mel bands.
    freq_min : scalar
        Minimum frequency for the first band.
    freq_max : scalar
        Maximum frequency for the last band.
    num_fft_bands : int
        Number of fft bands.
    center: bool
        (Default: True). If True, freq_min and freq_max indicate center frequencies of lowest and highest
        bands (not edges).

    Returns
    -------
    center_frequencies_mel : ndarray
    lower_edges_mel : ndarray
    upper_edges_mel : ndarray

    """
    if center:
        mel_min_center = hertz_to_mel(freq_min)
        mel_max_center = hertz_to_mel(freq_max)
        delta_mel = abs(mel_max_center - mel_min_center) / (num_bands - 1.0)
        frequencies_mel = append(mel_min_center - delta_mel, mel_min_center + delta_mel * arange(0, num_bands + 1))
        center_frequencies_mel = frequencies_mel[1:-1]  # Get even elements
        lower_edges_mel = frequencies_mel[:-2]
        upper_edges_mel = frequencies_mel[2:]
    else:
        mel_max = hertz_to_mel(freq_max)
        mel_min = hertz_to_mel(freq_min)
        delta_mel = abs(mel_max - mel_min) / (num_bands + 1.0)
        frequencies_mel = mel_min + delta_mel * arange(0, num_bands + 2)
        lower_edges_mel = frequencies_mel[:-2]
        upper_edges_mel = frequencies_mel[2:]
        center_frequencies_mel = frequencies_mel[1:-1]

    return center_frequencies_mel, lower_edges_mel, upper_edges_mel


def compute_mat(num_bands=12, freq_min=64, freq_max=8000,
                num_fft_bands=513, sample_rate=16000, scale='mel'):
    """Returns transformation matrix for mel spectrum or bark spectrum.

    Parameters
    ----------
    num_bands : int
        Number of mel/bark bands. Number of rows in mat.
        Default: 24
    freq_min : scalar
        Minimum frequency for the first band.
        Default: 64
    freq_max : scalar
        Maximum frequency for the last band.
        Default: 8000
    num_fft_bands : int
        Number of fft-frequency bands. This is NFFT/2+1 !
        number of columns in melmat/barkmat.
        Default: 513   (this means NFFT=1024)
    sample_rate : scalar
        Sample rate for the signals that will be used.
        Default: 44100
    scale : str
        Indicates scale ('bark'/'mel', default: 'mel')

    Returns
    -------
    mat : ndarray
        Transformation matrix for the mel/bark spectrum.
        Use this with fft spectra of num_fft_bands_bands length
        and multiply the spectrum with the melmat/barkmat
        this will tranform your fft-spectrum
        to a mel-spectrum/bark-spectrum.

    frequencies : tuple (ndarray <num_bands>, ndarray <num_fft_bands>)
        Center frequencies of the mel/bark bands, center frequencies of fft spectrum.

    """

    if scale == 'mel':
        center_frequencies_mel, lower_edges_mel, upper_edges_mel = \
            melfrequencies_mel_filterbank(
                num_bands,
                freq_min,
                freq_max,
                num_fft_bands,
                center=True
            )

        len_fft = float(num_fft_bands) / sample_rate
        center_frequencies_hz = mel_to_hertz(center_frequencies_mel)
        lower_edges_hz = mel_to_hertz(lower_edges_mel)
        upper_edges_hz = mel_to_hertz(upper_edges_mel)
        #freqs = linspace(0.0, sample_rate / 2.0, num_fft_bands)
        freqs = arange(0.0, sample_rate/2 + 1, sample_rate/num_fft_bands)
        melmat = zeros((num_bands, num_fft_bands))

        for imelband, (center, lower, upper) in enumerate(zip(
                center_frequencies_hz, lower_edges_hz, upper_edges_hz)):
            left_slope = (freqs >= lower) == (freqs <= center)
            melmat[imelband, left_slope] = (
                    (freqs[left_slope] - lower) / (center - lower)
            )

            right_slope = (freqs >= center) == (freqs <= upper)
            melmat[imelband, right_slope] = (
                    (upper - freqs[right_slope]) / (upper - center)
            )

        mat = melmat
        center_frequencies = center_frequencies_mel

    elif scale == 'bark':
        center_frequencies_bark, lower_edges_bark, upper_edges_bark = \
            barkfrequencies_bark_filterbank(
                num_bands,
                freq_min,
                freq_max,
                num_fft_bands,
                center=True
            )

        len_fft = float(num_fft_bands) / sample_rate
        center_frequencies_hz = bark_to_hertz(center_frequencies_bark)
        lower_edges_hz = bark_to_hertz(lower_edges_bark)
        upper_edges_hz = bark_to_hertz(upper_edges_bark)
        #freqs = linspace(0.0, sample_rate / 2.0, num_fft_bands)
        freqs = arange(0.0, sample_rate/2 + 1, sample_rate/num_fft_bands)
        barkmat = zeros((num_bands, len(freqs)))

        for ibarkband, (center, lower, upper) in enumerate(zip(
                center_frequencies_hz, lower_edges_hz, upper_edges_hz)):
            left_slope = (freqs >= lower) == (freqs <= center)
            barkmat[ibarkband, left_slope] = (
                    (freqs[left_slope] - lower) / (center - lower)
            )

            right_slope = (freqs >= center) == (freqs <= upper)
            barkmat[ibarkband, right_slope] = (
                    (upper - freqs[right_slope]) / (upper - center)
            )

        mat = barkmat
        center_frequencies = center_frequencies_bark

    return mat, (center_frequencies, freqs)


# def melfilterbank(num_mel_bands, fmin, fmax, nFFT, sr):
#     """Custom Mel Filter Bank Matrix computation.
#     Adapted from v_frq2bark.m (MATLAB function)
#         by Mike Brookes © 2006-2010.
#     Particularities:
#     - Converts to bark scale.
#     - fmin and fmax specify CENTER frequencies of lower and higher bands, NOT edges.
#     INPUTS:
#         num_mel_bands : number of mel frequency bands
#         fmin: center frequency of lower filter (in Hz)
#         fmax: center frequency of higher filter (in Hz)
#         nFFT: number of FFT points
#         sr: sample rate (Hz)
#     OUTPUTS:
#         FB: filter bank matrix
#         cf: center frequencies of filters
#     """
#     mflh = [fmin, fmax]
#     mflh = hertz_to_bark(mflh)  # Convert Hz to Bark scale
#     melrng = dot(mflh, linspace(-1, 1, 2))  # Bark range
#     melinc = melrng / (num_mel_bands - 1)
#     mflh = mflh + dot(linspace(-1, 1, 2), melinc)
#
#     # Center frequencies
#     cf = mflh[0] + arange(num_mel_bands + 2) * melinc  # Center frequencies in Bark including dummy ends
#     # Only the first point can be negative
#     for i in arange(len(cf))[1:]:
#         cf[i] = max(cf[i], 0)
#
#     # Convert center frequencies from Bark to Hz
#     mb = bark_to_hertz(cf)
