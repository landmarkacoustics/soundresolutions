# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np

def wial(before, after) -> float:
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

def b_and_a(X):
    for b, a in zip(X[:-1],X[1:]):
        yield b,a

class DFTSizeTester:
    def __init__(self,
                 fft_size,
                 n_overlaps : int = 50,
                 n_starts : int = 23,
                 window_name : str = 'hamming') -> None:
        Thing = sr.weighted_divergence_computer.Thing
        self._mech = sr.SpectrogramMachine(signal.get_window(window_name,
                                                               fft_size))
        self._thing = Thing(lambda X : (np.sum([wial(b,a) for b,a in b_and_a(X)])),
                            np.linspace(1,fft_size,n_overlaps,False,dtype=int),
                            n_starts)
        self._overlaps = self._thing.overlap_props(fft_size)
        self._buffer = np.zeros(self._thing.buffer_shape())

    def __call__(self, sound):
        self._thing(self._mech(sound,1),
                    self._buffer)
        return (self._overlaps[self._buffer[:,0].argmax()],
                self._buffer[:,0].max())
    
