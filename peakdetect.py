import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt
from scipy.interpolate import interp1d


def qint(ym1, y0, yp1):
    """(From MATLAB) QINT - quadratic interpolation of three adjacent samples.
    https://www.dsprelated.com/freebooks/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    Input parameters:
        y(-1) = ym1
        y(0) = y0
        y(1) = yp1
    Returns:
        p: extremum location
        y: height
        a: half-curvature
    Parabola is given by:
        y(x) = a*(x-p)^2+b"""
    p = (yp1-ym1)/(2*(2*y0-yp1-ym1))
    y = y0-0.25*(ym1-yp1)*p
    a = 0.5*(ym1-2*y0+yp1)
    return p, y, a


def peakdetect(r, Fs, thresh, Fmin):
    """Peak detection function. Adapted from Moliner et al. (2020).
    Input parameters:
        r = magnitude spectrum (absolute values) from STFT. Not expressed in dB (yet).
        Fs = sampling frequency.
        thresh = magnitude threshold (height)
        Fmin = minimum frequency for peak search range.
    Output parameters:
        exact = array of indices of exact peaks found in r.
        exact_peak = array of exact peak values, in dB."""
    rdB = 20*np.log10(r)
    s_win = (len(rdB)-1)*2
    # All local maxima in the magnitude spectrum are selected as possible peak candidates.
    locations, properties = find_peaks(rdB, height=thresh)
    peaks = properties['peak_heights']
    i = 0
    minpeak = Fmin
    minpeakbin = (minpeak/Fs)*s_win
    maxpeak = np.floor(Fs/2)  # Nyquist
    maxpeakbin = (maxpeak/Fs)*s_win
    for j in locations:
        if j < minpeakbin or j > maxpeakbin:
            peaks = np.delete(peaks, i)
            locations = np.delete(locations, i)
        else:
            i = i+1
    exact = locations.astype('float64')
    exact_peak = peaks
    i = 0
    if locations.size != 0:  # Checks if array isn't empty
        for j in locations:
            # For a more precise peak location, quadratic interpolation is used.
            ym1 = rdB[j-1]
            y0 = rdB[j]
            yp1 = rdB[j+1]
            p, y, a = qint(ym1, y0, yp1)
            exact[i] = j+p
            exact_peak[i] = y
            i = i+1
    return exact, exact_peak


def peakdetect1(signal, threshold=150):
    """Function that finds peaks (for instance in a spectrogram).
    A peak corresponds to a sample that is greater than its two nearest neighbors."""
    k = 2
    indices = []
    while k < len(signal) - 2:
        seg = np.asarray(signal[k - 2:k + 3])
        if np.max(seg) < threshold:
            k += 2
        else:
            if seg.argmax() == 2:  # Maximum is located in the center of the segment
                indices.append(k)
                k += 2
        k += 1
    return indices

