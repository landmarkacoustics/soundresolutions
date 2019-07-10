# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

#setup
sound = spek
Hz = 44100.0
Ny = Hz / 2
fft_size = 2048
step = int(.15*fft_size)
threshold = 0.005
#spaces
spk = SpectrumMachine(fft_size)
W = signal.hamming(fft_size)
Y = np.array([np.array(fft(W*sound[i-fft_size:i])) for i in range(fft_size,len(sound),step)])
spg = np.array([np.array(spk(W*sound[i-fft_size:i])) for i in range(fft_size,len(sound),step)])
#calculations
dphi = OrderedDict()
for k,v in zip(['dt','dw'],[0,1]):
    dphi[k] = np.diff(np.angle(Y)[:,:fft_size//2],axis=v) % np.pi

df = Hz / fft_size
dt = step / Hz
delays = dphi['dw'][1:,:] / (2*np.pi*df);
phases = dphi['dt'][:,1:] / (2*np.pi*dt)
T = np.arange(0,len(spg))*dt + dt
F = np.linspace(0,Ny,fft_size//2,False) + df
#plotting
plt.imshow(10*np.log10(spg.T),origin='lower',aspect='auto',cmap='gray_r',extent=[0,spg.shape[0]*step/Hz,0,Ny],interpolation='none')
for i, (d,p,y) in enumerate(zip(delays,phases,spg[1:,1:])):
    f = y > threshold
    plt.scatter(T[i+1]-d[f], (F[1:] - p)[f], color = 'yellow') and None

plt.show()
