# Copyright (C) 2018 by Landmark Acoustics LLC

def trill(time_vector: np.ndarray,
          times: Iterable[float],
          frequencies: Iterable[float],
          durations: Iterable[float],
          bandwidths: Iterable[float]) -> np.ndarray:
    r"""Create a series of chirps at the specified times.

    Parameters
    ----------
    time_vec : np.ndarray
        The time steps of the samples of some digital sound.
    times : Iterable[float]
        The center time of each element of the trill.
    frequencies : Iterable[float]
        The frequencies at the center of each element.
    durations : Iterable[float]
        The lengths, in time units, of each element.
    bandwidths :
        The height-, in frequency units, of each element.

    Returns
    -------
    out : np.ndarray
        The amplitude of the signal at each step of `time_vec`.

    See Also
    --------
    times : Generates `time_vec` arguments from a duration and sample
        rate.
    am_and_fm : used to produce each chirp.

    Examples
    --------
    This generates a series of twelve 100 ms tones that rise from 1 kHz
    to 8 kHz, have peak amplitudes at their halfway points, and are
    spaced 50 ms apart.

    >>> t = times(1800.0, 44.1)
    >>> T = np.arange(0, 12.0) * 150.0 + 75.0
    >>> F = [3.5] * 12
    >>> D = [100.0] * 12
    >>> B = [7] * 12
    >>> a = trill(t, T, F, D, B)
    >>> a.std(ddof=1)
    0.2545235077342643
    >>> np.sqrt(np.mean(a**2))
    0.25452190455601748

    """
    
    y = np.zeros(len(time_vector))
    
    for t, f, d, bw in zip(times, frequencies, durations, bandwidths):
        y += am_and_fm(time_vector, t, f, d, bw)
        
    return y


