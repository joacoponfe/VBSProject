import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt


def peakdetect0(r, Fs, thresh, Fmin):
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
    locations, properties = find_peaks(rdB, height=thresh)  # All local maxima in the magnitude spectrum are selected as possible peak candidates.
    peaks = properties['peak_heights']
    minpeak = Fmin
    minpeakbin = (minpeak/Fs)*s_win+1
    maxpeak = np.floor(Fs/2)  # Nyquist
    maxpeakbin = (minpeak/Fs)*s_win+1
    for j in locations:
        if j < minpeakbin or j > maxpeakbin:
            peaks[i] = []
            locations[i] = []
        else:
            i = i+1
    exact = locations
    exact_peak = peaks
    i = 1
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

