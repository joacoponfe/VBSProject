import numpy as np
from scipy.signal import medfilt, medfilt2d
from scipy.ndimage import median_filter

import sys


def transientness(X, nMedianH, nMedianV):
    # Median filtering
    [rows, columns] = np.shape(X)
    X_h_median = np.zeros((rows, columns))
    X_v_median = np.zeros((columns, rows))
    # Horizontal
    for i in np.arange(rows):
        X_h_median[i] = median_filter(abs(X[i]), nMedianH, mode='nearest')
    # Vertical
    for j in np.arange(columns):
        X_v_median[j] = median_filter(abs(X.transpose()[j]), nMedianV, mode='nearest')
    X_v_median = X_v_median.transpose()
    #X_v_median = median_filter(abs(X), nMedianV, mode='nearest')
    #X_v_median = medfilt(abs(X), nMedianV)
    # Horizontal
    #X_h_median = median_filter(abs(X), nMedianH, mode='nearest')
    #X_h_median = medfilt(abs(X), nMedianH)  # Qué onda el tema de la dimensión
    # Compute transientness
    Y = (abs(X_v_median)**2)/(abs(X_v_median)**2+abs(X_h_median)**2)
    Y = np.nan_to_num(Y)  # Convert NaN values to zeros (0)
    return Y
