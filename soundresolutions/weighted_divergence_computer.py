import numpy as np
from collections import OrderedDict
from typing import Any, Callable, List, Tuple

from .helpers import whole_steps
from .divergences import euclidean_distance as EuD
from .weights import coefficient_of_alienation as CoA

WDCFunc = Callable[[np.ndarray, np.ndarray], float]

class Thing:
    r"""Holy shit I have no idea...

    Parameters
    ----------
    F : WDCFunc
        The function that computes the weighted divergence between steps
    steps : Iterable[int]
        The steps that the resolution searcher will use
    starts : Iterable[int]
        The starts between each step

    """

    def __init__(self,
                 F : WDCFunc,
                 steps: List[int],
                 n_starts: int) -> None:
        self._F = F
        self._steps = np.array(steps, dtype = int)
        self._n_starts = n_starts


    def __call__(self, X: np.ndarray, B: np.ndarray = None) -> np.ndarray:
        r'''Summarizes the different possible divergences

        Parameters
        ----------
        X : np.ndarray
            The matrix to find divergences for
        B : np.ndarray [optional]
            A buffer for holding the means, variances, mins, and maxes

        Returns
        -------
        B : np.ndarray
            The mean, variance, min, and max for each step

        '''
        
        if B is None:
            B = np.tile(np.nan, self.buffer_shape())
        
        if B.shape != self.buffer_shape():
            raise ValueError('B should have shape {}, use `buffer_shape()` to find it!'.format(self.buffer_shape()))

        for i, step in enumerate(self._steps):
            B[i,:] = (0.0, 0.0, np.inf, -np.inf)
            starts = np.unique(np.linspace(0,step,self._n_starts,dtype=int))
            n = len(starts)
            for j, start in enumerate(starts):
                x = self._F(X[start::step,])
                B[i,0] += x
                B[i,1] += x*x
                B[i,2] = min(B[i,2],x)
                B[i,3] = max(B[i,3],x)
            if n > 1:
                B[i,1] = (B[i,1] - B[i,0]**2 / n) / (n - 1.0)
                B[i,0] /= n
            else:
                B[i,1] = 0.0

        return B


    def buffer_shape(self) -> tuple:
        r'''The shape that a buffer should have if used for `__call__`

        '''
        
        return (len(self._steps), 4)


    def overlap_props(self, maximum_steps: int) -> np.ndarray:
        r'''Converts steps to a proportional overlap, given max. steps possible

        Parameters
        ----------
        maximum_steps: int
            The width of the window that we're looking at diveregences across

        Returns
        -------
        out : np.ndarray
            Proportional overlap for each step

        '''
        
        return (1 - self._steps / maximum_steps)


class WeightedDivergenceComputer:
    """Computes weights, divergences, and weighted divergences between time steps of spectrograms"""
    def __init__(self, fft_size:int, n_steps:int=50, n_starts:int=7)->None:
        self._fft_size = fft_size
        self._starts = self.whole_steps(n_starts)
        self._steps = self.whole_steps(n_steps)
        self._values = np.tile(np.float(0), [len(self._steps),3])
    def FFTsize(self)->int:
        return self._fft_size
    def steps(self)->np.ndarray:
        return self._steps
    def overlaps(self)->np.ndarray:
        return 100 * (1 - self.steps() / self._fft_size)
    def starts(self)->np.ndarray:
        return self._starts
    def update(self, spg:np.ndarray,
               weight_func: WDCFunc,
               diverg_func: WDCFunc)->None:
        if spg.shape[0] < self.FFTsize():
            raise ValueError("too few time steps")
        self._values[:,:] = 0.0
        for start in self._starts:
            for i, step in enumerate(self._steps):
                for x,y in zip(spg[start + step::step,], spg[start:-step:step,]):
                    weight = weight_func(x,y)
                    diverg = diverg_func(x,y)
                    self._values[i,0] += weight
                    self._values[i,1] += diverg
                    self._values[i,2] += weight * diverg
        self._values /= len(self._starts)
    def values(self)->np.ndarray:
        return self._values
    def whole_steps(self, number:int)->np.ndarray:
        return whole_steps(1, self._fft_size, number)
    def weights(self)->np.ndarray:
        return self._values[:,0]
    def divergences(self)->np.ndarray:
        return self._values[:,1]
    def weighted_divergences(self)->np.ndarray:
        return np.sqrt(self._values[:,2]**2/self._fft_size)
    def summary(self) -> Tuple[float, float]:
        tmp = self.weighted_divergences()
        return (self.overlaps()[tmp.argmax()], tmp.max())

class BatchWDC:
    def __init__(self,
                 fft_sizes: np.ndarray = [64,128,256,512,1024,2048],
                 steps: int = 50,
                 starts: int = 23,
                 weight_func: WDCFunc = CoA,
                 diverg_func: WDCFunc = EuD)->None:
        self._workers = OrderedDict()
        for fft_size in fft_sizes:
            self._workers[fft_size] = WeightedDivergenceComputer(fft_size,
                                                                 steps,
                                                                 starts)
        self._wf = weight_func
        self._df = diverg_func
        self._answers = np.zeros([steps, len(self._workers)])
    def __call__(self, spectrogram) -> np.ndarray:
        for i, (fft_size, M) in enumerate(self._workers.items()):
            M.update(spectrogram, self._wf, self._df)
            self._answers[:,i] = M.weighted_divergences()
        return self._answers
