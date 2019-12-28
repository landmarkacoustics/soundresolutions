# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np
from . import SpaceRanger


class Synthesizer(SpaceRanger):
    """Syntactic sugar for creating waveforms with a `SpaceRanger`

    Parameters
    ----------
    variables: dict
        The parameters that will vary among outputs. Should be iterables.
    constants: dict
        The parameters that are the same for every output. Should be constants.

    Yields
    ------
    out : Tuple[Dict, np.ndarray]
        The parameters that define a sound, and the corresponding waveform.

    See Also
    --------
    SpaceRanger : the parent class

    Examples
    --------
    >>> moog = Synthesizer(dict(duration=[.1,.2,.3]),
    ...                    dict(Hz = 44100))
    >>> for D,X in moog:
    ...     print(len(Synthesizer.times(D['duration'],D['Hz'])))
    ...
    4410
    8820
    13230

    """

    def __init__(self, variables: dict, constants: dict) -> None:
        super().__init__(self.synthesize, constants, variables)

    def times(duration: float, Hz: float) -> np.ndarray:
        """Array of times from [-duration/2,duration/2) with interval 1/Hz

        Parameters
        ----------
        duration : float
            The length, in time units, of the output array
        Hz : float
            The number of samples per time unit

        Returns
        -------
        out : np.ndarray
            Linearly increasing values centered on 0.0

        Examples
        --------
        >>> Synthesizer.times(0.5, 8)
        array([-0.25 , -0.125,  0.   ,  0.125])

        """

        return np.linspace(-duration/2,
                           duration/2,
                           int(duration*Hz),
                           False)

    @classmethod
    def synthesize(cls, duration: float, Hz: float) -> np.ndarray:
        """A placeholder for demonstration purposes

        Parameters
        ----------
        duration : float
            The length, in time units, of the output array
        Hz : float
            The number of samples per time unit

        Returns
        -------
        out : np.ndarray
            An array of zeros of length int(duration * Hz)

        Examples
        --------
        moog = Synthesizer(dict(),dict(Hz=8,duration=0.5))
        >>> np.array([x for D,x in moog])
        array([[ 0.,  0.,  0.,  0.]])

        """

        return np.zeros(int(duration*Hz))
