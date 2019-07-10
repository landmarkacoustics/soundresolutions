# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import soundresolutions as sres

Thing = sres.weighted_divergence_computer.Thing

Hz = 44100.0
Ny = Hz / 2
fft_sizes = 2**np.arange(6,12)

slopes = np.array(10**np.linspace(2,6,50),dtype=int)
duration = 0.2
durations = np.array([min(duration, Ny/m) for m in slopes])
bandwiths = slopes * durations

m = sres.SpectrogramMachine(signal.hamming(256))
for i, (d,b) in enumerate(zip(durations[ix],bandwiths[ix])):
    plt.subplot(3,3,i+1)
    sound = sres.Whistles.synthesize(Hz,d,Ny/2,b,1e-15,2)
    spg = m(sound,0.8)
    plt.imshow(10*np.log10(spg.T), origin='lower',aspect='auto',extent=[0,d,0,Ny],interpolation='none') and None
    plt.xlim([0,duration]) and None

plt.show()

al = lambda x: np.sum(np.abs(np.diff(x)))
r = lambda x,y: np.corrcoef(x,y)[0,1]
tial = lambda x, y : al(x) * (1 - r(x,y)**2)

all_tials = lambda X : np.array([tial(y,x) for x,y in zip(X[:-1],X[1:])])


things = OrderedDict()
bros = OrderedDict()
mechs = OrderedDict()
for fft_size in fft_sizes:
   things[fft_size] = sres.weighted_divergence_computer.Thing(lambda X : np.sum(all_tials(X)), np.linspace(1,fft_size,50,False,dtype=int), 23)
   bros[fft_size] = np.zeros(things[fft_size].buffer_shape())
   mechs[fft_size] = sres.SpectrogramMachine(signal.hamming(fft_size))

for th, b, m in zip(things.values(), bros.values(), mechs.values()):
    th(m(sound,1),b).shape
    
for (fft_size, th), b in zip(things.items(), bros.values()):
    plt.plot(th.overlap_props(fft_size), b[:,0], label = fft_size)

plt.legend(loc='best')
plt.show()

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
        self._mech = sres.SpectrogramMachine(signal.get_window(window_name,
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
    
