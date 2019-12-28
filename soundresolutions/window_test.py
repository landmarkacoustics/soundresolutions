# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np
from collections import OrderedDict
from typing import Callable, List

from .wdc import WDC, WDCFunc, BatchWDC
from .wiv import wiv

all_window_names = [
    'barthann',
    'bartlett',
    'blackman',
    'blackmanharris',
    'bohman',
    'boxcar',
    'flattop',
    'hamming',
    'hann',
    'nuttall',
    'parzen'
]


class WindowTester:
    def __init__(self,
                 fft_sizes: np.ndarray = 2**np.arange(6, 12),
                 divergence_function: WDCFunc = wiv,
                 steps: int = 59,
                 starts: int = 29,
                 windownames: List[str] = all_window_names):
        r'''An environment for comparing different window functions.

        Parameters
        ----------
        fft_sizes : np.ndarray, optional
            The window sizes to try for each window.
        divergence_function : WDCFunc, optional
            The function for computing divergences between adjacent windows.
        steps : int, optional
            The number of different overlaps to compare.
        starts : int, optional
            The (maximum) number of offsets at each number of overlaps.
        windownames : List[str], optional
            The names of each window function.

        See Also
        --------
        soundresolutions.wdc : The module that computes divergences.
        signal.get_window : The function that computes the window shapes.

        '''

        self._windownames = windownames
        self._wdcs = OrderedDict()
        for fft_size in fft_sizes:
            self._wdcs[fft_size] = WDC(fft_size=fft_size,
                                       function=divergence_function)

        self._batch = BatchWDC(self._wdcs, starts, np.arange(0, 1, 1/steps))
        self._results = dict()
        self._results = np.zeros([len(windownames),
                                  len(fft_sizes)],
                                 dtype=float)

    def __call__(self, soundmaker: Callable[[None], np.ndarray]) -> None:
        r'''Finds each window`s best overlap for the sound(s) from `soundmaker`

        Parameters
        ----------
        soundmaker : Callable[[None], np.ndarray]
            Probably a lambda expression.

        Returns
        -------
        nuthin'

        See Also
        --------
        calculated_overlaps : returns the stored results from this function.

        '''

        overlaps = self._batch._overlaps
        wdc_view = self._wdcs.items()

        for i, w in enumerate(self._windownames):
            for wdc in self._wdcs.values():
                wdc.set_window(w)

            self._batch(soundmaker())
            sums = self._batch.sums().T
            for j, ((fft_size, wdc), x) in enumerate(zip(wdc_view, sums)):
                steps = wdc.n_steps(overlaps[np.argmax(x)])
                self._results[i, j] = 1.0 - steps / fft_size

    def calculated_overlaps(self) -> np.ndarray:
        r''' Accessor for the results of calling the object.

        Returns
        -------
        results : a 2D np.ndarray containing the results for each combination
            of window and FFT size.

        '''

        return self._results


if __name__ == '__main__':
    import numpy as np
    import soundresolutions as sr

    N = 10000

    FS = 2 ** np.arange(6, 12)

    window_tester = WindowTester(fft_sizes=FS)

    window_tester(lambda: np.random.normal(0, 1, N))

    print(f'{"Window":^18}'+''.join([f'{x:^10}' for x in FS])+f'{"Mean":^10}')
    Oh = window_tester.calculated_overlaps()
    for w, o, m in zip(sr.all_window_names, Oh, Oh.mean(1)):
        print(f'{w:>18}' +
              ''.join([f'{x:10.1f}' for x in 100*o]) +
              f'{100*m:10.1f}')
