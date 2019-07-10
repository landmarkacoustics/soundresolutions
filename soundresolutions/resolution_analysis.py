# Copyright (C) 2018 Landmark Acoustics, LLC

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import importlib
import os
os.chdir("/Users/stern/Documents/Python")
import soundresolutions
from soundresolutions import WeightedDivergenceComputer as WDC
from soundresolutions import spectrogram

tau = np.pi * 2

duration = 0.2
sample_rate = 44100

N = int(duration * sample_rate)
time_vector = np.linspace(0, duration, N, False)

fft_sizes = np.array(2**np.arange(6,12), dtype = int)
n_steps = 50
n_starts = 7

computers = dict()

for fft_size in fft_sizes:
    computers[fft_size] = WDC(fft_size, n_steps, n_starts)

window_function = "hamming"
pitches = soundresolutions.whole_steps(1/duration, sample_rate /  2, 6)

saws = np.tile(0.0, [N, len(pitches)])
for j, phi in enumerate(pitches):
    saws[:,j] = signal.sawtooth(tau * phi * time_vector)

saw_wds = np.tile(0.0, [len(fft_sizes), len(pitches), n_steps])
for i, fft_size in enumerate(fft_sizes):
    comp = computers[fft_size]
    W = signal.get_window(window_function, fft_size)
    print(fft_size)
    for j, X in enumerate(saws.T):
        comp.update(spectrogram(X,W))
        saw_wds[i,j,:] = comp.weighted_divergences()
        print(j)

chirps = np.tile(0.0, [N, len(pitches)])
starting_pitch = sample_rate // 4
for j, phi in enumerate(pitches):
    chirps[:,j] = signal.chirp(time_vector, starting_pitch, duration, phi)

chirp_wds = np.tile(0.0, [len(fft_sizes), len(pitches), n_steps])
for i, fft_size in enumerate(fft_sizes):
    comp = computers[fft_size]
    W = signal.get_window(window_function, fft_size)
    print(fft_size)
    for j, X in enumerate(chirps.T):
        comp.update(spectrogram(X,W))
        chirp_wds[i,j,:] = comp.weighted_divergences()
        print(j)

k = 50
n = N // k
t = (np.arange(n) - n//2) / Hz
pulse = dict(starts=np.linspace(0,N,k,False,dtype=int),
             frequencies=np.random.normal(11025,11025/10,k),
             bandwidths=np.random.normal(0.5,0.05,k))
for ix, phi, p in zip(pulse['starts'], pulse['frequencies'], pulse['bandwidths']):
    train[ix:ix+n] = soundresolutions.scale_to_unit_energy(signal.gausspulse(t,fc=phi,bw=p))

p = soundresolutions.proportions_from_snr(5)
train = p[0] * train + p[1] * soundresolutions.scale_to_unit_energy(soundresolutions.audio.white_noise(N))
train_wds = np.tile(0.0, [len(fft_sizes), n_steps])

for i, fft_size in enumerate(fft_sizes):
    comp = computers[fft_size]
    W = signal.get_window(window_function, fft_size)
    print(fft_size)
    comp.update(spectrogram(train,W))
    train_wds[i,:] = comp.weighted_divergences()
