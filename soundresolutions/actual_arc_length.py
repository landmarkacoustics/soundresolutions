# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

def harmcomb(taus,N,f0):
    ampl = np.zeros(len(taus))
    for ef in np.arange(f0,N/2,f0):
        ampl += np.sin(taus*ef)
    return ampl

N = 44100
fft_sizes = 2**np.arange(6,12)

for K in fft_sizes:
    df = N/K
    taus = 2*np.pi*np.arange(-K/N/2,K/N/2,1/N)
    W = signal.gaussian(K,K/5)
    fundaments = np.linspace(N/K/2,N/2,50,False)
    als = np.zeros(len(fundaments))
    for i, f0 in enumerate(fundaments):
        ampl = harmcomb(taus,N,f0)*W
        spek = np.square(np.abs(np.fft.rfft(ampl)))/K
        als[i] = np.sqrt(np.mean(spek))

    plt.plot(fundaments, als, label = K)

plt.legend(loc='best')
plt.show()
