import numpy as np
from scipy.stats import pearsonr

def coefficient_of_alienation(x: np.ndarray, y: np.ndarray) -> np.float:
    r"""Calculates 1 - R**2

    Parameters
    ----------
    x : np.ndarray
        Real-valued, one-dimensional
    y : np.ndarray
        Also real-value, one-dimensional

    Returns
    -------
    out : float
        The amount of variation in `x` that is not explained by `y`.

    See Also
    --------
    numpy.corrcoef : can also calculate correlation coefficients

    """
    
    return 1 - pearsonr(x,y)[0]**2

