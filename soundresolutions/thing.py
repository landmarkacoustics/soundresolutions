# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np
import inspect
from typing import Any, Callable, Iterable, List, Set, Tuple

from .helpers import proportions_from_snr
from .audio import white_noise

class BackgroundNoiseThing:
    def __init__(self,SNR):
        self._a, self._b = proportions_from_snr(SNR)
    def __call__(self, sound:np.ndarray)->np.ndarray:
        return self._a * sound + self._b * white_noise(len(sound))

def is_iterable(x)->bool:
    return hasattr(x,'__iter__')

def disgorge(*args):
    if args:
        if is_iterable(args[0]):
            head = disgorge(*args[0])
        else:
            head = [args[0]]
        return head + disgorge(*args[1:])
    else:
        return []

def all_the_things(*args):
    N = len(args)
    counts = [len(x) for x in args]
    ix = [0] * N
    yield [x[i] for i,x in zip(ix,args)]
    should_increase = True
    for i in range(N):
        if should_increase:
            ix[i] += 1
            if ix[i] >= counts[i]:
                ix[i] = 0
                should_increase = True
            else:
                should_increase = False

def combined_things(**kwargs):
    keys = list(kwargs.keys())
    N = len(keys)
    vals = kwargs.values()
    counts = [len(v) for v in vals]
    indices = [0] * N
    while indices[0] < counts[0]:
        D = dict()
        for i,k,v in zip(indices, keys, vals):
            D.update({k:v[i]})
        yield D
        indices[-1] += 1
        for i in range(N-1, 0, -1):
            if indices[i] == counts[i]:
                indices[i-1] += 1
                indices[i] = 0

class SpaceRanger:
    def __init__(self,
                 function: Callable,
                 static_parameters: dict,
                 varied_parameters: dict) -> None:
        self._F = function
        actual_keys = set(list(static_parameters) + list(varied_parameters))
        missing = self.required_args() - actual_keys
        if missing:
            raise ValueError('missing arguments: {}'.format(missing))
        extras = actual_keys - self.required_args()
        filt = lambda d : dict([(k,v) for k,v in d.items() if k not in extras])
        self._static = filt(static_parameters)
        self._varying = filt(varied_parameters)
    def __iter__(self) -> Iterable[Tuple[dict,Any]]:
        if self._varying:
            for D in combined_things(**self._varying):
                D.update(self._static)
                yield (D, self._F(**D))
        elif self._static:
            yield (self._static, self._F(**self._static))
    def required_args(self) -> Set[str]:
        return set(inspect.getfullargspec(self._F).args) - {'self'}

class Synthesizer(SpaceRanger):
    def __init__(self, variables: dict, constants: dict) -> None:
        super().__init__(self.synthesize, constants, variables)
    def _desired_arguments(self) -> Set[str]:
        return set()
    def times(duration: float, Hz: float) -> np.ndarray:
        return np.linspace(-duration/2,
                           duration /2,
                           int(duration*Hz),
                           False)
    def synthesize(self) -> np.ndarray:
        return None
