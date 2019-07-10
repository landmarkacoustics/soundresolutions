# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np
from scipy import signal

from typing import Tuple


duration = 0.1
order = 3
epsilon = .06
peak = 1-epsilon
degree = 0.9
k = np.log(epsilon/peak) / max(times)**degree
func = lambda x : peak * np.exp(k*x**degree)
#func = lambda x : np.array([11000,6000]) - 50000.0*x
#func = lambda x : .6 - 4 * x
interpolations = 14
plt.figure()
plt.suptitle(interpolations)
times = np.linspace(0,duration,interpolations)
states = dict(A = np.zeros((interpolations, order)),
              C = np.zeros((interpolations, order)),
              D = np.zeros(interpolations))

for i,t in enumerate(times):
    b, a = signal.butter(order, func(t))
    A, B, C, D = signal.tf2ss(b,a)
    states['A'][i,:] = A[0,:]
    states['C'][i,:] = C[0,:]
    states['D'][i] = D[0].333

fits = dict(A = np.zeros((interpolations, order)),
            C = np.zeros((interpolations, order)))
for j in range(order):
    for key in  fits.keys():
        fits[key][:,j] = np.polyfit(times,
                                    states[key][:,j],
                                    interpolations - 1)

fits['D'] = np.polyfit(times, states['D'], interpolations - 1)

x = np.linspace(0,duration,101)
for i,key in enumerate(('A','C')):
    plt.subplot(1,3,i+1)
    plt.title(key)
    for j, c in enumerate(['b','g','r','c','y','m','k'][:min(order,7)]):
        plt.plot(x, np.polyval(fits[key][:,j],x), c + '-')
        plt.plot(times, states[key][:,j], c + '.')

plt.subplot(1,3,3)
plt.title('D')
plt.plot(x, np.polyval(fits['D'],x), 'k-')
plt.plot(times, states['D'], 'k.')

#plt.show()
plt.figure()

states['A'] = make_a(order)
states['B'] = make_b(order)
states['C'] = make_c(order)
states['D'] = make_d(order)

states['x'] = np.zeros((order,1))

times = np.linspace(0,duration,int(duration*44100))
X = np.random.normal(0,1,len(times))
Y = np.zeros(len(times))

for i, (t,x) in enumerate(zip(times,X)):
    states['A'][0,:] = [np.polyval(p,t) for p in fits['A'].T]
    states['C'][:] = [np.polyval(p,t) for p in fits['C'].T]
    states['D'][0] = np.polyval(fits['D'],t)
    Y[i] = np.matmul(states['C'],states['x']) + states['D'] * x
    states['x'] = np.matmul(states['A'], states['x']) + states['B'] * x

from fourier import SpectrogramMachine

M = SpectrogramMachine(np.hamming(512))

for i, z in enumerate((X,Y)):
    plt.subplot(1,2,1+i)
    S = 10*np.log10(M(z,205))
    plt.imshow(S.T,origin='lower',aspect='auto',cmap='gray_r')

plt.show()

def make_a(order: int) -> np.ndarray:
    A = np.zeros((order, order))
    A[1:,:-1] = np.eye(order - 1)
    return A

def make_b(order: int) -> np.ndarray:
    return np.concatenate([[1],np.zeros(order-1)]).reshape((order,1))

def make_c(order: int) -> np.ndarray:
    return np.zeros(order)

def make_d(order: int) -> np.ndarray:
    return np.array([1.0])

def state_space(order: int) -> Tuple[np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray]:
    return (make_a(order),
            make_b(order),
            make_c(order),
            make_d(order))

