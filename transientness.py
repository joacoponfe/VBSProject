import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import median_filter
import sys


def transientness(X, nMedianH, nMedianV):
    # Median filtering
    # Vertical
    X_v_median = median_filter(abs(X), nMedianV, mode='nearest')
    #X_v_median = medfilt(abs(X), nMedianV)
    # Horizontal
    X_h_median = median_filter(abs(X), nMedianH, mode='nearest')
    #X_h_median = medfilt(abs(X), nMedianH)  # Qué onda el tema de la dimensión
    # Compute transientness
    Y = (abs(X_v_median)**2)/(abs(X_v_median)**2+abs(X_h_median)**2)
    Y = np.nan_to_num(Y)  # Convert NaN values to zeros (0)
    return Y
