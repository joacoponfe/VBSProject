import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import median_filter


def transientness(X, nMedianH, nMedianV):
    # Median filtering
    # Vertical
    X_v_median = median_filter(abs(X), nMedianV)
    # Horizontal
    X_h_median = median_filter(abs(X), nMedianH)
    # Compute transientness
    Y = (X_v_median**2)/(X_v_median**2+X_h_median**2)
    Y[np.isnan(Y)] = 0
    return Y
