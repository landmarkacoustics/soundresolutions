# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np

def wial(before, after) -> float:
    r'''Weighted Independent Arc Length between vectors

    Multiplies the arc length of `after` by the one minus the squared
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
        The arc length of `after` times the coefficent of alienation between
        `before` and `after`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-.1,.1,201)
    >>> wial(np.exp(-x**2/5), np.exp(-(x-.05)**2/5))
    0.0039321647408345518

    '''
    
    partials = np.zeros([3,2])
    for i, x in enumerate([before, before*after, after]):
        partials[i,0] = np.sum(x)
        if i != 1:
            partials[i,1] = np.sum(x**2)

    n = len(before)
    euk = np.sum(np.abs(np.diff(after)))
    
    num = (n*partials[1,0] - partials[0,0]*partials[2,0])
    den = (n*partials[0,1] - partials[0,0]**2) * (n*partials[2,1] - partials[2,0]**2)
    
    return euk * (1 - num**2 / den)
