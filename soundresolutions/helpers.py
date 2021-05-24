# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np


def autocorrelation(x: np.ndarray, lag: int = 1) -> float:
    r'''Computes the Pearson autocorrelation of `x`, offset by `lag`.

    Parameters
    ----------
    x : np.ndarray
        A numeric vector of at least length `lag` + 2
    lag : int
        An integer in [0, len(`x`) - 2].

    Returns
    -------
    xc : float
        The Pearson product-moment autocorrelation coefficient

    See Also
    --------
    pearson : the Pearson product-moment correlation coefficent between arrays.

    Examples
    --------
    >>> np.random.seed(42)
    >>> x = np.random.normal(0,1,20)
    >>> autocorrelation(x,5)
    0.29108118622217644

    '''

    if len(x) - lag < 2:
        raise ValueError('Too short for autocorrelation')
    return pearson(x[:-lag or None], x[lag:])


def pearson(X: np.ndarray, Y: np.ndarray) -> float:
    r'''Uses one pass to find the Pearson product-moment correlation

    This is the naive, numerically unstable approach

    Parameters
    ----------
    X : np.ndarray
        an array of real numbers
    Y : np.ndarray
        an array of real numbers

    Returns
    -------
    out : float
        The Pearson product-moment correlation coefficient between the arrays

    '''

    if len(X) != len(Y):
        raise ValueError('both arguments must have same length')
    if len(X) < 2:
        raise ValueError('the arrays must have at least 2 values each')

    sx = 0.0
    sy = 0.0
    sxy = 0.0
    sxx = 0.0
    syy = 0.0

    n = len(X)

    for x, y in zip(X, Y):
        sx += x
        sy += y
        sxy += x * y
        sxx += x * x
        syy += y * y

    num = n*sxy - sx*sy
    den = np.sqrt((n*sxx - sx**2) * (n*syy - sy**2))

    return num / den


def proportions_from_snr(snr: np.float) -> tuple:
    r'''calculates the proportional eneregy of signal and noise, given an SNR'''
    k = snr / (1 + snr)
    return (k, 1-k)


def pSNR_from_dbSNR(snr: np.float) -> float:
    r'''A noise's proportional amplitude from its decibel signal-to-noise ratio.

    '''

    return 10**(-snr/10)


def rms(x: np.ndarray) -> np.float:
    r'''calculates the Root-Mean-Square of an array'''
    return np.sqrt(np.nanmean(np.square(x)))


def scale_to_unit_energy(x: np.ndarray) -> np.ndarray:
    r'''divides an array by the Root-Mean-Square of its values'''
    return x / rms(x)


def whole_steps(start: int, stop: int, number: int) -> np.ndarray:
    r'''interpolates up to number steps between start and stop.

    Each step is an integer.

    '''

    dif = abs(stop - start)
    if dif < number:
        number = dif
    return np.array(np.linspace(start, stop, number, False).round(),
                    dtype=int)
