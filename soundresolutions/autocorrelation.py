# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np

def autocorrelation(x: np.ndarray) -> np.ndarray:
    r'''Finds the autocorrelations of `x` with itself.

    Parameters
    ----------
    x : np.ndarray
        Any data sample you want the autocorrelation of
    
    Returns
    -------
    out : np.ndarray
        The Pearson product-moment correlation of `x` with itself. It will be
    the same length as `x`. In other words, the overlaps are in (0%,100%].
    The magnitude of the autocorrelation trails off at low overlaps because
    the overlapping portions are zero-padded to maintain a constant length.

    See Also
    --------
    scipy.signal.correlate : does the lion`s share of the computations

    Examples
    --------
    find the autocorrelation of a wicked quick impulse
    >>> autocorrelation([0,0,-1,1,0,0])
    array([ 0. ,  0. ,  0. ,  0. , -0.5,  1. ])

    '''

    tmp = np.array(x,dtype=float)
    tmp -= tmp.mean()
    n = len(x)
    return (signal.correlate(tmp,tmp)[:n]/tmp.var()/n)

if __name__ == '__main__':
    
    import numpy as np
    from scipy import signal
    import soundresolutions as sr
    
    for w in sr.window_test.all_window_names:
        x = signal.get_window(w,N)
        X = sr.autocorrelation(x)[N//2:]
        i = np.argmin(X**2)
        print(f'{w:^20}{i:>7}{100.0*(0.5+i/N):>7.2f}')
