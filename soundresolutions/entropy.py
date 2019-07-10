# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import soundresolutions as sr

N = 44100
duration = 0.2
time = np.arange(-duration/2, duration/2, 1/N)
hum = lambda f0: signal.sawtooth(f0*2*np.pi*time)

fundaments = np.array([10,20,50,100,200,500])
n_overlaps = 25
fft_sizes = 2**np.arange(6,12)

def scaleH(x):
    p = x / np.nansum(x)
    return -np.nansum(p*np.log(p))/np.log(len(x))

for i, f0 in enumerate(fundaments):
    plt.subplot(2,3,i+1)
    hmmm = hum(f0)
    for K in fft_sizes:
        x = np.zeros(n)
        oves = np.linspace(0,K,n,False,dtype=int)
        for j, o in enumerate(oves):
            f,t,S = signal.spectrogram(hmmm, N, ('gaussian',K/5), K, o, scaling = 'spectrum')
            x[j] = np.sum(np.array([np.sum(np.abs(np.diff(x))) for x in S.T]))
        
        plt.plot(oves/K, x, label = K)
    
    plt.title(f'$F_0=${f0}')
    plt.legend(loc='best')

plt.show()

def lengthalong(x):
    return np.sum(np.abs(np.diff(x)))

def mi(x,y):
    return (lengthalong(x) + lengthalong(y)) / np.sum(np.abs(x-y))
