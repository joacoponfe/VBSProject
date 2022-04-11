def ola_norm_coef(win_analysis, win_synthesis, nHop_synthesis):
    """Defines OLA normalized coefficient
    Adapted from Moliner et al. (2020)
    Input parameters:
        win_analysis: analysis window
        win_synthesis: synthesis window
        nHop_synthesis: synthesis hop size
    Returns:
        y: OLA coefficient
    """
    nWin = len(win_analysis)
    nHop = nHop_synthesis

    win = win_analysis * win_synthesis
    idx = int(nWin/2)
    y = win[idx]

    m = 1
    i = idx-m*nHop
    j = idx+m*nHop

    while i > 0 and j <= nWin:
        y = y+win[i]+win[j]
        m = m+1
        i = idx-m*nHop
        j = idx+m*nHop

    return y


