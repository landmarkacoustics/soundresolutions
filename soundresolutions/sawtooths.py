# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np
from scipy import signal

from . import gauss_am, TAU, Synthesizer


class Sawtooths(Synthesizer):
    """Produces sawtooth waveforms across a parameter space

    Parameters
    ----------
    variables: dict
        A dictionary of iterables whose keys correspond to parameters.
    constants: dict, optional
        The input parameters that do not vary across outputs.

    Yields
    ------
    parameters, waveform: (dict, np.ndarray)
        The parameters passed to generate the waveform, and the sound.
    """
    def __init__(self,
                 variables: dict,
                 constants: dict = {
                     'Hz': 44100.0,
                     'decay_eps': 1e-15,
                     'decay_degree': 2
                     }):
        super().__init__(variables, constants)

    def suggested_fft_size(fundamental: float,
                           Hz: float) -> int:
        """Finds a suggested FFT size for describing a harmonic stack.

        Parameters
        ----------
        fundamental : float
            The fundamental frequency of the harmonic stack
        Hz : float
            The sample rate of the sound

        Returns
        -------
        out : int
            The FFT size that is the next-greatest power of two

        Examples
        --------
        >>> Sawtooths.suggested_fft_size(300, 44100)
        512

        """

        return int(2**(1+np.ceil(np.log2(Hz/fundamental))))

    @classmethod
    def synthesize(cls,
                   fundamental,
                   duration,
                   Hz,
                   decay_eps,
                   decay_degree) -> np.ndarray:
        """ A sawtooth waveform with a gaussian amplitude envelope.

        Parameters
        ----------
        Hz : float
            The sample rate, in time units, of the output sound.
        fundamental : float
            The fundamental frequency of the harmonic stack.
        duration : float
            The time spanned, in time units, of the output sound.
        decay_eps : float
            The value that the amplitude envelope decreases below.
        decay_degree : float
            The abruptness with which the envelope drops off at its ends.

        Returns
        -------
        out : np.ndarray
            A sawtooth waveform sampled at `Hz` samples per time unit.

        Examples
        --------

        """

        t = Synthesizer.times(duration, Hz)
        return (gauss_am(t, duration, decay_eps, decay_degree)
                * signal.sawtooth(fundamental * TAU * t))


# def best_for(sound: np.ndarray, machines, offsets, buffer) -> tuple:
#     best_score = 0.0
#     best_fft = 0
#     best_overlap = 0.0

#     for i, ((fft_size, SM), O) in enumerate(zip(machines.items(), offsets)):
#         tmp = mieds(SM(sound,1), O, buffer)
#         m = tmp.max()
#         if m > best_score:
#             best_score = m
#             best_fft = fft_size
#             best_overlap = 100 * (1 - O[tmp.argmax()]/fft_size)

#     return (best_fft, best_overlap, best_score)

# from matplotlib import pyplot as plt
# Hz = 44100.0
# Ny = Hz / 2
# duration = 0.05
# times = np.linspace(0,duration,int(duration*Hz),False)
# n_offsets = 50
# B = np.tile(0.0, [len(times),n_offsets])
# fft_sizes = 2**np.arange(6,12)
# O = np.tile(0,[len(fft_sizes), n_offsets])
# machines = OrderedDict()
# for i, fft_size in enumerate(fft_sizes):
#     machines[fft_size] = sres.SpectrogramMachine(signal.hamming(fft_size))
#     O[i,:] = np.linspace(1,fft_size,n_offsets)

# n = 20
# centers = Ny / 2**np.arange(12,5,-1)
# deltas = np.diff(centers)/2
# fzeros = np.zeros(n*len(fft_sizes))
# for i, (lo,hi) in enumerate(zip(centers[1:]-deltas,centers[1:]+deltas)):
#     fzeros[n*i:n*(i+1)] = np.linspace(lo,hi,n,True,dtype=int)

# i = np.zeros(len(fzeros),dtype=int)
# f = np.zeros(len(fzeros),dtype=float)
# hork = np.core.records.fromarrays([fzeros,i,f,f,f],
#                                   names='F0,FFT size,Overlap,TIED,RMS')
# del i,f
# for i in range(len(hork)):
#     s = signal.sawtooth(hork[i]['F0']*2*np.pi*times) +
#                         np.random.normal(0,1/101,len(times))

#     hork[i]['RMS'] = np.sqrt(np.nanmean(np.square(s)))
#     hork[i]['FFT size'], hork[i]['Overlap'], hork[i]['TIED'] = \
#          best_for(s,machines,O,B)
#     if i % 10 == 0:
#         print(i)

# coldict = dict([*zip(fft_sizes,['b','g','r','c','m','y'])])
# plt.scatter(hork['F0'], hork['TIED'],
#             s=100, c=[coldict[x] for x in hork['FFT size']])
# plt.xscale('log')
# plt.ylim([0,max(hork['TIED'])])
# plt.vlines(centers[1:],0.0,max(hork['TIED']))
# plt.show()

# tmp = hork['TIED']/hork['RMS']
# plt.scatter(hork['F0'],tmp,s=100,c=[coldict[x] for x in hork['FFT size']])
# plt.ylim([0,max(tmp)])
# plt.vlines(centers[1:],0.0,max(tmp))
# plt.xlabel('Fundamental Frequency of Sawtooth (Hz)')
# plt.ylabel('Mean Weighted Independent Energy Distance (unitless)')
# plt.xlim(hork['F0'][[0,-1]])
# plt.xscale('log')
# del tmp
# plt.show()
