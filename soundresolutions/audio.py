# Copyright (C) 2018 by Landmark Acoustics LLC
""" Functions for creating audio signal data."""

import numpy as np


TAU = 2.0 * np.pi


def am_tone(time_vec: np.ndarray,
            carrier_freq: float,
            am_freq: float,
            am_amp: float = 1.0,
            phase: float = np.pi) -> np.ndarray:
    r"""Create an amplitude-modulated sine wave.

    Parameters
    ----------
    time_vec : np.ndarray
        The time steps of the samples of some digital sound.
    carrier_freq : float
        The frequency of the tone that will be amplitude modulated.
    am_freq : float
        The frequency at which the amplitude is modulated.
    am_amp : float, optional
        The magnitude of the modulation, relative to the carrier's
        amplitude.
    phase : float, optional
        The phase of the modulation, relative to the carrier's phase.

    Returns
    -------
    out : np.ndarray
        An array that contains the amplitude at each time point
        specified in `time_vec`.

    See Also
    --------
    times : a function for generating `time_vec` arguments from a
        duration and sample rate.

    Examples
    --------
    This generates 0.1s of a 4 kHz signal modulated at 20 Hz.

    >>> t = times(100.0, 44.1)
    >>> a = am_tone(t, 4, 0.02)
    >>> a.std(ddof=1)
    0.4362056644673169

    """

    carrier = np.sin(TAU * time_vec * carrier_freq)
    modulator = am_amp * np.cos(TAU * time_vec * am_freq + phase)
    output = (1 + modulator) * carrier
    return output / max(abs(output))


def gauss_am(time_vector: float,
             duration: float,
             decay_eps: float = 1e-15,
             decay_degree: float = 2) -> np.ndarray:
    r"""find the decay rate parameter for a sound that decays to
    `epsilon` in `duration`

    Parameters
    ----------
    duration : float
        The length, in time units, of the the noise
    epsilon : float, optional
        The value that the amplitude will decay to during `duration`
    degree : int, optional
        The power that the term is raised to.

    Returns
    -------
    out : np.ndarray
        The decay rate parameter.

    Examples
    --------
    This finds the decay rate to go from 1.0 to 2**-15 in 100 ms

    >>> decay_rate(100)
    0.004158883083359672

    """

    decay_rate = 2**decay_degree * np.log(decay_eps) / duration ** decay_degree
    return np.exp(decay_rate * time_vector ** decay_degree)


def times(duration: float = 1,
          sample_rate: float = 44100) -> np.ndarray:
    r"""Create the time steps of the samples of some digital sound.

    The time steps will be in the interval [0, `duration`) and the
    length of the output will be `sample_rate` * `duration`.

    The units of `duration` should be compatible with those of
    `sample_rate`. If, for example, you specify `duration` as 300ms and
    `sample_rate` as 44.1 kHz, you'll get the same answer as specifying
    a `duration` of 0.3s and a `sample_rate` of 44100 Hz.

    Parameters
    ----------
    duration : float, optional
        The length, in time units, of the sound being sampled.
    sample_rate : float, optional
        The number of samples per time unit of the sound being sampled.

    Returns
    -------
    out : np.ndarray
        An array that contains the time locations of each sample. It
        will start at 0.0. Its elements will be separated by
        1.0 / `sample_rate`. Its length will be the integer closest to
        `sample_rate` * `duration` items.

    See Also
    --------
    numpy.arange : `times` is just syntactic sugar for a call to arange.

    Examples
    --------
    This generates the times for 100 milliseconds of CD-quality sound:

    >>> t = times(100.0, 44.1)
    >>> len(t)
    4410
    >>> print(t[0])
    0.0
    >>> print(t[-1])
    99.977324263

    """

    return np.arange(duration, step=1/sample_rate)


def white_noise(*args, **kwargs) -> np.ndarray:
    r"""Create an array of white noise drawn from the standard normal.

    This is just an array of standard normally-distributed values,
    divided by the maximum of its absolute values. You can specify the
    length of the array directly, or with a duration and sample rate.

    Parameters
    ----------
    n : int, optional
        The length, in samples, of the the noise
    duration : float, optional
        The length, in time units, of the noise
    sample_rate : float, optional
        The number of samples per time unit of the noise

    Returns
    -------
    out : np.ndarray
        An array of random values, also known as white noise.

    See Also
    --------
    numpy.random.normal : generates normally-distributed random numbers.

    Examples
    --------
    This generates 100 ms of CD-quality white noise:

    >>> np.random.seed(42)
    >>> t = audio.times(100.0,44.1)
    >>> a = audio.white_noise(len(t))
    >>> a.min()
    -0.82554027097127247
    >>> a.max()
    1.0

    """

    n = 0

    if kwargs:
        if 'n' in kwargs:
            n = int(kwargs['n'])
        elif 'duration' in kwargs and 'sample_rate' in kwargs:
            n = int(kwargs['duration']*kwargs['sample_rate'])
        else:
            raise ValueError('bad keyword arguments')
    elif args:
        if len(args) == 1:
            n = int(args[0])
        elif len(args) == 2:
            n = int(args[0] * args[1])
        else:
            raise ValueError('bad positional arguments')

    if n <= 0:
        raise ValueError('the result would have no samples!')
    return np.random.normal(0, 1, n)
