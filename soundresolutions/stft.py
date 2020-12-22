# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np
from scipy import fftpack


def stft(x: np.ndarray, b: np.ndarray = None) -> np.ndarray:
    r'''The short-term Fourier transform of `x`, optionally stored in `b`.

    This function formats the output of a call to `scipy.fftpack.rfft`. The
    exact format that is returned depends upon whether `b` is real- or complex-
    valued, and, if `b` is real-valued, whether the `len(b)` is `len(x)` or
    `1 + len(x)//2`. See the description of `b` for details.

    Parameters
    ----------
    x : np.ndarray
        This vector must be real-valued
    b : np.ndarray, optional
        The buffer where the data will be output, if supplied. It may be real-
    or complex-valued. If b is complex-valued, then it must have length `1 +
    len(x)//2`. If b is real-valued, then it may have length `1 + len(x)//2`,
    in which case it will be filled with the real spectrum of `x`, or it may
    have length `len(x)`, in which case it will be filled with the output of
    `scipy.fftpack.rfft(x)`.

    Returns
    -------
    b : np.ndarray
        This will be the complex-valued STFT of `x` if `b` was None. Otherwise,
    it will be None.

    Raises
    ------
    ValueError
        If `b` has an invalid length.

    See Also
    --------
    scipy.fftpack.rfft : The function that calculates the Fourier transform.

    Notes
    -----
    The outputs are not scaled in any way (just like the outputs from `rfft`).

    Examples
    --------
    >>> import numpy as np
    >>> N = 4
    >>> foo = np.cos(2*np.pi*np.linspace(0,1,N,False))
    >>> np.round(stft(foo))
    array([-0.+0.j,  2.-0.j,  0.+0.j])
    >>> b = np.zeros(N)
    >>> stft(foo,b);np.round(b)
    array([-0.,  2., -0.,  0.])
    >>> n = 1 + N//2
    >>> b = np.zeros(n,dtype=complex)
    >>> stft(foo,b);np.round(b)
    array([-0.+0.j,  2.-0.j,  0.+0.j])
    >>> b = np.zeros(n)
    >>> stft(foo,b);np.round(b)
    array([ 0.,  4.,  0.])

    '''

    returnb = False
    real_length = 1 + len(x)//2

    if b is None:
        b = np.zeros(real_length, dtype=complex)
        returnb = True

    q = fftpack.rfft(x)
    real_indices = slice(1, None, 2)
    imag_indices = slice(2, None, 2)

    if b.dtype is np.dtype(float):
        if len(b) == len(x):
            b[:] = q
        elif len(b) == real_length:
            b[0] = q[0]**2
            b[1:] = q[real_indices]**2
            b[1:-1] += q[imag_indices]**2
        else:
            raise ValueError('what kind of weird buffer length is that?')
    elif b.dtype is np.dtype(complex):
        if len(b) != real_length:
            raise ValueError('what kind of weird buffer length is that?')

        b[0] = q[0] + 0j
        b.real[1:] = q[real_indices]
        b.imag[1:-1] = q[imag_indices]
        b.imag[-1] = 0.0

    if returnb:
        return b
