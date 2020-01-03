# Copyright (C) 2018 by Landmark Acoustics LLC

from inspect import getfullargspec
from itertools import islice
from typing import Any, Callable, Dict, Iterable, Set, Tuple


def combined_things(**kwargs) -> Dict[Any, Any]:
    """A generator over every combination of the supplied arguments.

    Each value in `kwargs` should be an iterable. Builds a dict
    with every key in `kwargs` for each possible combination of the
    elements of each argument.

    Parameters
    ----------
    kwargs : dict
        Each member's value should be an iterable.

        The values do not have to have the same length. If there is only
        one item for a key, it should still be passed as an iterable
        of some kind, rather than a scalar or other constant.

    Yields
    ------
    out : dict
        A dict object with the same keys as `kwargs`, but with only one
        item for each value

    Examples
    --------
    Work through the six combinations possible from the inputs.

    >>> a = ['a']
    >>> pi = [3, 1, 4]
    >>> b = {True, False}

    >>> for i, D in enumerate(combined_things(letter = a,
                                              number = pi,
                                              logical = b)):
        print('{}:    {}'.format(i,D))

    0:    {'logical': False, 'letter': 'a', 'number': 3}
    1:    {'logical': False, 'letter': 'a', 'number': 1}
    2:    {'logical': False, 'letter': 'a', 'number': 4}
    3:    {'logical': True, 'letter': 'a', 'number': 3}
    4:    {'logical': True, 'letter': 'a', 'number': 1}
    5:    {'logical': True, 'letter': 'a', 'number': 4}
    """

    keys = list(kwargs.keys())
    N = len(keys)
    vals = kwargs.values()
    counts = [len(v) for v in vals]
    indices = [0] * N
    while indices[0] < counts[0]:
        D = dict()
        for i, k, v in zip(indices, keys, vals):
            tmp = next(islice(v, i, None))
            D.update({k: tmp})
        yield D
        indices[-1] += 1
        for i in range(N-1, 0, -1):
            if indices[i] == counts[i]:
                indices[i-1] += 1
                indices[i] = 0


class SpaceRanger:
    """ A function to evaluate across a parameter space and the space itself

    Parameters
    ----------
    function : Callable
        Must have parameters with the same names as the keys in the other args
    static_parameters: dict
        A dictionary of constants, rather than iterables
    varied_parameters: dict
        A dictionary of iterables

    Yields
    ------
    out : Tuple[Dict, Any]
        A tuple with the specific values passed to `function` and its result.

    Examples
    --------
    >>> pi = [3, 1, 4]
    >>> b = {True, False}
    >>> foo = lambda t, b, x: str(b) + ':' + str(x) + t
    >>> buzz = SpaceRanger(foo, dict(t='a'), dict(b=b, x=pi))
    >>> for d,v in buzz:
        print(f'{d}\t{v}')

    {'logical': False, 'letter': 'a', 'number': 3}	False:3a
    {'logical': False, 'letter': 'a', 'number': 1}	False:1a
    {'logical': False, 'letter': 'a', 'number': 4}	False:4a
    {'logical': True, 'letter': 'a', 'number': 3}	True:3a
    {'logical': True, 'letter': 'a', 'number': 1}	True:1a
    {'logical': True, 'letter': 'a', 'number': 4}	True:4a

    """

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

        def filt(d):
            return dict([(k, v) for k, v in d.items() if k not in extras])

        self._static = filt(static_parameters)
        self._varying = filt(varied_parameters)

    def __iter__(self) -> Iterable[Tuple[dict, Any]]:
        if self._varying:
            for D in combined_things(**self._varying):
                D.update(self._static)
                self._update(D)
                yield (D, self._F(**D))
        elif self._static:
            yield (self._static, self._F(**self._static))

    def __len__(self) -> int:
        p = 1
        for v in self._varying.values():
            p *= len(v)
        return p

    def required_args(self) -> Set[str]:
        """Reports the names of the arguments to `function`

        Returns
        -------
        out : Set[str]
            a set of argument names

        Examples
        --------
        >>> pi = [3, 1, 4]
        >>> b = {True, False}
        >>> foo = lambda t, b, x: str(b) + ':' + str(x) + t
        >>> buzz = SpaceRanger(foo, dict(t='a'), dict(b=b, x=pi))
        >>> buzz.required_args()
        {'t', 'b', 'x'}

        """

        return set(getfullargspec(self._F).args) - {'self'}

    def _update(self, D: Dict) -> None:
        """Override this to change state variables during iteration

        Parameters
        ----------
        D : Dict
            Arguments that would be passed to `self._F`

        Returns
        -------
        """

        pass
