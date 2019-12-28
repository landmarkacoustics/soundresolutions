# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np


def wiss(before, after) -> float:
    r'''Weighted Independent Sum Of Squares between vectors

    Multiplies the sum of squares of `after` by the one minus the squared
    correlation between `before` and `after`.

    Parameters
    ----------
    before : np.ndarray
        A vector describing the previous instant.
    after : np.ndarray
        A vector describing the current instant.

    Returns
    -------
    out : float
        The SS of `after` times the coefficent of alienation between
        `before` and `after`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-.1,.1,201)
    >>> wiss(np.exp(-x**2/5), np.exp(-(x-.05)**2/5))
    0.00026975516695114442

    '''

    partials = np.zeros([3, 2])
    for i, x in enumerate([before, before*after, after]):
        partials[i, 0] = np.sum(x)
        if i != 1:
            partials[i, 1] = np.sum(x**2)

    n = len(before)
    var = (n*partials[2, 1] - partials[2, 0]**2)

    num = (n*partials[1, 0] - partials[0, 0]*partials[2, 0])
    den = (n*partials[0, 1] - partials[0, 0]**2) * var

    return (1 - num**2 / den) * var / n
