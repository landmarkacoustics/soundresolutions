# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from collections import OrderedDict

def full_xcorr(x:np.ndarray, y:np.ndarray = None)->np.ndarray:
    if y is None:
        y = x
        n = min(len(x),len(y))-1
        xc = np.zeros(n-1)
    for i in range(1,n):
        xc[i-1] = np.corrcoef(x[i:],y[:-i])[0,1]
        
    return xc

window_names = [
    'barthann',
    'bartlett',
    'blackman',
    'blackmanharris',
    'bohman',
    'boxcar',
    'flattop',
    'hamming',
    'hann',
    'nuttall',
    'parzen'
]

windows = OrderedDict([(n,signal.get_window(n,1000)) for n in window_names])

auto_overlaps = 100 - np.array([np.argmin(full_xcorr(v)**2) for v in windows.values()])/10

fft_sizes = 2**np.arange(6,12)

tial_overlaps = np.array([
    [78.1, 78.1, 77.7, 77.9, 77.9, 75.9],
    [75.0, 75.7, 75.7, 75.9, 75.9, 75.9],
    [81.2, 82.0, 82.0, 82.0, 81.9, 81.9],
    [85.9, 83.5, 83.9, 83.9, 83.9, 83.9],
    [82.8, 82.0, 82.0, 82.0, 81.9, 81.9],
    [98.4, 99.2, 99.6, 99.8, 99.9, 99.9],
    [90.6, 89.8, 89.8, 89.8, 89.9, 89.9],
    [78.1, 75.7, 75.7, 75.9, 75.9, 75.9],
    [79.6, 78.1, 77.7, 77.9, 77.9, 77.9],
    [82.8, 83.5, 83.9, 83.9, 83.9, 83.9],
    [82.8, 83.5, 83.9, 83.9, 83.9, 83.9]
])

