import numpy as np


def NLD(x, h):
    """NLD process signal x with transfer function h."""
    hN = len(h)
    for n in np.arange(hN):
        if n == 0:
            y = h[n]  # For n = 0, x^0 = 1
        else:
            y = y + h[n] * (x ** n)
    return y


