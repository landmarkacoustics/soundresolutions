# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np
from scipy import signal
from .helpers import pSNR_from_dbSNR, rms

def dirac_comb(start: float,
               end: float,
               pulse_rate: float,
               sample_rate: int)->np.ndarray:
    r"""A series of energy spikes at the specified rate.

    Each spike is a single sample with value 1. Their separation is determined
    by `pulse_rate` and `sample_rate`. The duration of the output is determined
    by the separation of `start` and `end` and the `sample_rate`. The phase of
    the comb is set so that there is a spike at index 0.

    Parameters
    ----------
    start: float
        The time (in time units) at the beginning of the comb.
    end : float
        The time (in time units) at the end of the comb.
    pulse_rate : float
        The number of pulses per time rate of the comb.
    sample_rate : int
        The samples per unit time of the comb.

    Returns
    -------
    comb : np.ndarray
        A time series like a Dirac comb, except with positive and negative values at each spike.

    Examples
    --------
    >>> dirac_comb(-4,4,1/3,1)
    array([ 0,  1, -1,  0,  1, -1,  0,  1])

    Notes
    -----
    It might be more pythonic to do:
    [0 if x % sample_rate/pulse_rate else 1 for x in range(start*sample_rate,end*sample_rate)]
    but it's way the hell slower

    """

    lo = int(min(start,end) * sample_rate)
    step = int(sample_rate / pulse_rate)
    origin = (step - lo) % step
    comb = np.zeros(int(np.ceil(np.abs(end - start) * sample_rate)), dtype = int)
    comb[origin::step] = 1
    return comb

def pulse_train(duration : float,
                rate : float,
                center_freq : float,
                duty_cycle : float = 1/7,
                snr : float = 20.0,
                sample_rate : float = 44100.0) -> np.ndarray:
    r"""Creates a series of gaussian pulses like an echolocation buzz.

    Parameters
    ----------
    duration : float
        The duration of the entire train.
    rate : float
        The number of pulses per time, in the units of duration.
    center_freq : float
        The central frequency of the pulse.
    duty_cycle : float [optional]
        The proportion of each period that has pulse energy.
    snr : float [optional]
        The signal-to-noise ratio, in decibels
    sample_rate : float [optional]
        The sample rate of the sound

    Returns
    -------
    wave : np.ndarray
        The waveform of the pulse train

    Examples
    --------
    TBD

    See Also
    --------
    signal.gausspulse : used to make each pulse in the train

    """

    npulses = int(duration * rate)
    
    start_time = -0.5 * duration
    end_time = 0.5 * duration
    offset = end_time/npulses

    times = np.linspace(start_time, end_time, duration*sample_rate, False)

    wave = np.random.normal(0, pSNR_from_dbSNR(snr), len(times))

    pband = find_bandwidth(duty_cycle/rate, center_freq)
    
    for t in np.linspace(start_time + offset,
                         end_time + offset,
                         npulses, False):
        wave += signal.gausspulse(t=times-t,
                                  fc=center_freq,
                                  bw=pband)
    
    return wave


def find_bandwidth(duration: float,
                   center_freq: float) -> float:
    """Computes the bandwidth, as a proportion of the Nyquist, of a pulse.

    Parameters
    ----------
    duration : float
    The duration of the pulse.

    center_freq: float
    The center frequency of the pulse.

    Returns
    -------
    out : float
    The bandwidth of the pulse, as a proportion of the Nyquist frequency.

    Examples
    --------
    >>> find_bandwidth(0.02, 11025)
    0.006306791461182619

    See Also
    --------
    pulse_train : uses this functino to create a series of pulses
    signal.gausspulse : the function that makes individual pulses

    """
    
    return 1 / (0.71908948 * duration * center_freq)




