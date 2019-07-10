# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

def pole_params(order: float = 2, band: np.ndarray = [.1,.2]) -> np.ndarray:
    p = signal.butter(order, band, 'bandpass', False, 'zpk')[1]
    return(np.concatenate([np.unique(p.real), np.unique(p.imag)]))

p = np.array([pole_params(band=[l,l+.1]) for l in np.arange(0.1,.5,.01)])

for i, x in enumerate(p.T):
    plt.plot(x,label=i)

plt.legend(loc='best')
plt.show()
