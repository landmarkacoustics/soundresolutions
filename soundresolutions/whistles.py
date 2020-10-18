# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np

from scipy.signal import sweep_poly

from . import gauss_am

from . import Synthesizer


class Whistles(Synthesizer):
    """Emit one frequency-modulated pulse for each combination of parameters.

    Parameters
    ----------
    variables: dict
        The parameters that will vary among outputs. Should be iterables.
    constants: dict
        The parameters that are the same for every output. Should be constants.

    Yields
    ------
    out: Tuple[Dict, np.ndarray]
        The parameters that define a FM pulse, and the corresponding waveform.

    See Also
    --------
    Synthesizer: the parent class, obvs
    Whistles.synthesize: The waveform maker. Defines the required parameters.
    signal.sweep_poly: Computes the sinusoid.
    soundresolutions.gauss_am: Computes the amplitude envelope.

    Examples
    --------
    >>> from soundresolutions import rms, Whistles
    >>> sample_rate = 44100
    >>> flipper = Whistles(dict(duration=[.1,.2,.3]),
    ...                    dict(Hz = sample_rate,
    ...                    center_frequency = sample_rate / 4
    ...                    bandwidth = sample_rate / 4,
    ...                    decay_eps = 1e-15,
    ...                    decay_degree = 2))
    for D, X in flipper:
    ... d = D['duration']
    ... r = rms(X)
    ... n = len(X)
    ... print(f'duration = {:>.3}, RMS = {:<.5}, N = {:>5}'.format(d,r,n))

    duration = 0.1, RMS = 0.32654, N =  4410
    duration = 0.2, RMS = 0.32654, N =  8820
    duration = 0.3, RMS = 0.32654, N = 13230

    """

    def _update(self,
                D: dict) -> None:
        r"""Updates the center slope state.

        Users should not call this method

        Parameters
        ----------
        D: dict
            contains center_frequency, duration, and bandwidth

        """

        p = Whistles.polynomial_coefficients(D['center_frequency'],
                                             D['duration'],
                                             D['bandwidth'])

        self._center_slope = p[2]

    def __init__(self,
                 variables: dict,
                 constants: dict = {
                     'Hz': 44100.0,
                     'decay_eps': 1e-15,
                     'decay_degree': 2
                 }) -> None:
        super().__init__(variables, constants)
        self._center_slope = np.nan

    def center_slope(self) -> float:
        r"""The central (max. abs. mag.) slope of the most recent whistle

        """

        return self._center_slope

    def polynomial_coefficients(cls,
                                center_frequency: float,
                                duration: float,
                                bandwidth: float) -> np.ndarray:
        r"""Computes the coefficients of an S-shaped cubic polynomial.

        Parameters
        ----------
        center_frequency: float
            The time at the center of the curve.
        duration: float
            The length, in time units, of the curve.
        bandwidth: float
            The height, in frequency units, of the curve.

        Returns
        -------
        out: np.ndarray
            The four terms of a cubic polynomial, from the 3rd to the 0th.

        Examples
        --------
        This finds the coefficients for a 100 ms sweep from 1 kHz to 8 kHz.

        >>> swoop(3.5, 100, 7)
        array([ -1.40000000e-05,   0.00000000e+00,   1.05000000e-01,
                 3.50000000e+00])

        """

        return np.array([-2.0*bandwidth / duration**3,
                         0.0,
                         1.5*bandwidth / duration,
                         center_frequency])

    @classmethod
    def synthesize(cls,
                   Hz: float,
                   duration: float,
                   center_frequency: float,
                   bandwidth: float,
                   decay_eps: float,
                   decay_degree: float) -> np.ndarray:
        r"""Create a chirp with an S-shape in the frequency domain.

        Parameters
        ----------
        Hz: float
            The sample rate of the output waveform
        duration: float
            The length, in time units, of the curve.
        center_frequency: float
            The frequency at the center of the curve
        bandwidth: float
            The height, in frequency units, of the curve.

        Returns
        -------
        out: np.ndarray
            The amplitude of the signal at each step of `time_vec`.

        See Also
        --------
        Synthesizer.times: time instants to go with the waveform
        gauss_am: the whistle's amplitude envelope.
        Whistles.polynomial_coefficients: for the S-shaped frequency trace

        Examples
        --------
        This generates a 100 ms tone that rises from 1 kHz to 8 kHz and has
        peak amplitude at 50 ms into the sound.

        >>> a = Whistles.synthesize(44.1, 100, 3.5, 7, 1e-15, 2)
        >>> a.std(ddof=1)
        0.31175974612191376

        """

        t = Synthesizer.times(duration, Hz)
        p = Whistles.polynomial_coefficients(center_frequency,
                                             duration,
                                             bandwidth)

        return (gauss_am(t, duration, decay_eps, decay_degree)
                * sweep_poly(t, p))
