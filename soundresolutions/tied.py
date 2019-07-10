# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np


def tied(X: np.ndarray, Y:np.ndarray) -> float:
    r'''Total Independent Energy Distance

    This is the naive, numerically unstable approach

    Parameters
    ----------
    X : np.ndarray
        an array of real numbers
    Y : np.ndarray
        an array of real numbers

    Returns
    -------
    out : float
        The Euclidean distance between X and Y, weighted by their
        coefficient of alienation, which is 1 - R**2

    '''
    
    if len(X) != len(Y):
        raise ValueError('both arguments must have same length')

    sx = X.sum()
    sy = Y.sum()
    sxy = (X*Y).sum()
    sxx = np.square(X).sum()
    syy = np.square(Y).sum()
    
    n = len(X)
    
    num = (n*sxy - sx*sy)**2
    den = (n*sxx - sx**2) * (n*syy - sy**2)
    
    return np.sqrt(sxx + syy - 2*sxy)*(1-num/den)


def mieds(X: np.ndarray,
          overlaps: np.ndarray,
          buff: np.ndarray = None) -> np.ndarray:
    r'''Finds the average TIED for each possible overlap

    Parameters
    ----------
    X : np.ndarray
        Usually, this will be a spectrogram
    overlaps : np.ndarray
        Should be an array of integer step sizes
    buff : np.ndarray
        Lets you reuse memory to be more efficient!

    Returns
    ----------
    out : np.ndarray
        The mean independent energy distances from X for each overlap

    '''

    dims = (X.shape[0], len(overlaps))
    
    if buff is None:
        buff = np.tile(np.nan, dims)
    elif buff.shape != dims:
        raise ValueError('buff\'s dims don\'t match X and overlaps')

    for j, o in enumerate(overlaps):
        for i in range(o, dims[0]):
            buff[i,j] = tied(X[i,:],X[i-o,:])

    return np.nansum(buff,0) / overlaps

def best_address(X: np.ndarray) -> np.ndarray:
    return np.array(X.argmax() / np.array(X.shape[None:None:-1]),dtype=int)

def best_mied(address: np.ndarray, fft_sizes: np.ndarray, overlaps: np.ndarray) -> tuple:
    r'''Find the fft size and overlap for an address'''

    best_fft = fft_sizes[address[0]]
    best_overlap = overlaps[address[0],address[1]]
    return (best_fft, 100 * (1-best_overlap/best_fft))

def foo(X: np.ndarray, overlap: int, proportion: float) -> float:
    total = 0
    stops = np.arange(0,overlap,max(1,int((1-proportion)*overlap)))
    for start in stops:
        tmp = X[start:,]
        total += sum([tied(x,y) for x,y in zip(tmp[:-overlap:overlap],tmp[overlap::overlap])])
    
    return total / len(stops)


# subfist = OrderedDict()
# for i, k in enumerate(string.ascii_lowercase[:7]):
#     snd = Sound('/home/ben/Documents/Sounds/soundresolutions/physeter {}.wav'.format(k)).read_frames()
#     n = len(snd)
#     subfist[k] = dict(time=np.linspace(0,n/Hz,n,False),sound=snd, spgs=OrderedDict(), mieds=OrderedDict())
#     B = np.tile(np.nan,(n, n_overlaps))
#     plt.subplot(2,4,i+1)
#     plt.title(k)
#     for fft_size, m in machines.items():
#         x = m(subfist[k]['sound'],1)
#         subfist[k]['spgs'][fft_size] = x
#         o = lo(fft_size)
#         ied = mieds(x, o, B)
#         subfist[k]['mieds'][fft_size] = ied
#         plt.plot(1-o/fft_size, ied, label=fft_size)
    
#     plt.xlabel('Overlap (% of Window Length)')
#     plt.ylabel('MIED (V)')

# plt.legend(loc='best')
# plt.show()
