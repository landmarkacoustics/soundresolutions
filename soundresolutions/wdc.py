# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np
from scipy.signal import get_window
from typing import Callable, Tuple
from collections import OrderedDict

from . import scan_by_steps, real_spk, wial

WDCFunc = Callable[[np.ndarray, np.ndarray], float]


class WDC:
    r'''Summarizes the weighted divergences between spectra for one resolution.

    Parameters
    ----------
    fft_size: int
        The size of the window for the stft.
    window_name: str, optional
        The window function that will be applied at each STFT.
    function: Callable[[np.ndarray, np.ndarray], float], optional
        The function for computing weighted divergences between spectra.

    See Also
    --------
    soundresolutions.wial: Defaults to the Weighted Independent Arc Length.

    '''

    def __init__(self,
                 fft_size: int,
                 window_name: str = 'hamming',
                 function: WDCFunc = wial) -> None:
        self._fft_size = fft_size
        self.set_window(window_name)
        self._stft_buffer = np.zeros([1 + fft_size//2, 2])
        self._time_buffer = np.zeros(self._fft_size)
        self._wdc_func = function

    def set_window(self, window_name: str) -> None:
        r'''Pre-calculates the window function for the STFT.

        Parameters
        ----------
        window_name: str
            The name of the function according to `get_window`.

        See Also
        --------
        scipy.signal.get_window: Does the actual work.
        '''

        self._window = get_window(window_name, self._fft_size, False)

    def n_steps(self, overlap: float) -> int:
        r'''Finds the number of steps that corresponds to the window overlap

        Parameters
        ----------
        overlap: float
            This should be in [0,1). You deserve the weird results if it isn't!

        Returns
        -------
        n_steps: int
            The number of samples between adjacent windows, given the overlap.

        Examples
        --------
        >>> foo = WDC(10)
        >>> foo.n_steps(.9)
        1

        '''

        return max(1, int(round(self._fft_size * (1.0-overlap))))

    def __call__(self,
                 sound: np.ndarray,
                 overlap: float) -> Tuple[float, float, int]:
        r'''Finds statistics of the divergence function over the sound.

        The calculated statistics are mean, variance, and sample size.

        Parameters
        ----------
        sound: np.ndarray
            The signal; it should be read into memory in its entirety.
        overlap: float
            Proportional overlap between adjacent windows. Must be in [0,1).

        Returns
        -------
        sum: float
            The total of the divergences at the requested resolution.
        variance: float
            The sample variance of the divergences across the sound.
        count: int
            The number of steps performed at the requested resolution.

        See Also
        --------
        soundresolutions.real_spk: Real Fourier transforms for the spectra.

        '''

        n_steps = self.n_steps(overlap)
        self._stft_buffer[:] = 0.0
        S = 0.0
        SS = 0.0
        n = 0
        gen = scan_by_steps(sound, self._time_buffer, n_steps)
        for i, x in enumerate(gen):
            now = i % 2
            then = 1 - now
            self._stft_buffer[:, now] = real_spk(self._window * x)
            w = self._wdc_func(self._stft_buffer[:, then],
                               self._stft_buffer[:, now])
            if np.isfinite(w):
                S += w
                SS += w*w
                n += 1

        return (S, (n * SS - S**2) / (n**2 - n), n)


class BatchWDC:
    r'''runs several WDC analyses at once'''

    def __init__(self, wdcs, n_starts, overlaps):
        self._wdcs = wdcs
        self._n_starts = n_starts
        self.set_overlaps(overlaps)

    def set_overlaps(self, overlaps):
        self._overlaps = overlaps
        self._S = np.zeros([len(self._overlaps), len(self._wdcs)])
        self._V = self._S.copy()
        self._N = self._S.copy()

    def __call__(self, sound):
        self._S[:] = 0.0
        self._V[:] = 0.0
        self._N[:] = 0.0
        for j, (k, wdc) in enumerate(self._wdcs.items()):
            for i, o in enumerate(self._overlaps):
                starts = np.linspace(0, wdc.n_steps(o), self._n_starts, False)
                starts = np.array(np.unique(np.around(starts)), dtype=int)
                starts.sort()
                d = 1.0 / len(starts)
                for s in starts:
                    m, v, n = wdc(sound[s:], o)
                    self._S[i, j] += m * n
                    self._V[i, j] += v * n
                    self._N[i, j] += n

                self._S[i, j] *= d
                self._V[i, j] *= d
                self._N[i, j] *= d

    def sums(self):
        return self._S / self._N

    def variances(self):
        return self._V / self._N

    def counts(self):
        return self._N

    def best_sums(self):
        bests = self.sums()
        return ([*self._wdcs.keys()][np.argmax(np.max(bests, 0))],
                self._overlaps[np.argmax(np.max(bests, 1))])


def some_wdcs(fft_sizes: np.ndarray) -> OrderedDict:
    wdcs = OrderedDict()
    for fft_size in fft_sizes:
        wdcs[fft_size] = WDC(fft_size)
    return wdcs

# for fft_size, y, n in zip(fft_sizes, S.T, W.T):
#    plt.plot(overlaps, y/n, label = fft_size)
#
# plt.legend(loc='best')
# plt.show()
