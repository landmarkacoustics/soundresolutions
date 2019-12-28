# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np
from scipy.fftpack import fft
from scipy.signal import hamming
from matplotlib import pyplot as plt


def as_pdf(x: np.ndarray) -> np.ndarray:
    return x / np.nansum(x)


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    if len(p) != len(q) or any(q <= 0):
        return np.nan
    f = (p > 0)
    return np.nansum(p[f] * (np.log(p[f]) - np.log(q[f])))


def spk(x: np.ndarray) -> np.ndarray:
    fft_size = len(x)
    half = fft_size//2
    tmp = fft(x)[:half]
    return (tmp.real**2 + tmp.imag**2)/fft_size


# from soundfile import Soundfile as Sound
# soundfilename = "~/Desktop/one_chirp.wav.bak"
# chirp = Sound(soundfilename)

# dirname = "/home/ben/Documents/Landmark Acoustics/Clients/Hitchcock/VOT/"

# pooh = Sound(dirname + "C3Pooh2Log2_high135.29.wav")

# boo = Sound(dirname + "C3Boo13Log8_high95.59.wav")

# dirname = "/home/ben/Documents/Landmark Acoustics/" \
#           "Technology Development/Optimal Overlap/

# swsp = Sound(dirname + "three_swsp_syllables.wav")

# syll = Sound(dirname + "one_swsp_syllable.wav")

# du = Sound(dirname + "Deutschish/du.wav")

# N = du.frames()
# Hz = du.samplerate()
# X = du.read_frames(N)

# saus = dict()
# maus = dict()
# sums = dict()
# frange = dict()
# trange = dict()
# klaus = dict()
# Y = dict()

# for fft_size in 2**np.arange(6,12):
#     half_fft = fft_size // 2
#     W = hamming(fft_size)
#     ix = np.arange(fft_size)
#     n = N - fft_size
#     Y[fft_size] = np.zeros((n,half_fft))
#     frange[fft_size] = np.array([0,Hz/2])
#     trange[fft_size] = (np.array([fft_size,N])-half_fft)/Hz
#     for i in range(n):
#         Y[fft_size][i,] = spk(W*X[i+ix])

#     pdf_cache = np.zeros((fft_size, half_fft))
#     pdf_cache[-1,:] = 1/half_fft

#     klaus[fft_size] = np.zeros((n,fft_size))

#     for offset in range(min(fft_size,n)):
#         for time_index in range(offset):
#             pdf_cache[time_index,] = as_pdf(Y[fft_size][time_index,])
#             klaus[fft_size][time_index,offset] = \
#                 kl_div(pdf_cache[time_index,],
#                        pdf_cache[-1,])
#         for time_index in range(offset, n):
#             cache_index = time_index % fft_size
#             pdf_cache[cache_index,] = as_pdf(Y[fft_size][time_index,])
#             klaus[fft_size][time_index,offset] = \
#                 kl_div(pdf_cache[cache_index,],
#                        pdf_cache[(time_index - offset - 1) % fft_size,])

#     sums[fft_size] = np.nanmean(Y[fft_size],1)
#     ovlps = np.arange(1,fft_size+1)
#     maus[fft_size] = np.nansum(klaus[fft_size],
#                                0)/ovlps
#     saus[fft_size] = np.nansum(klaus[fft_size]*sums[fft_size][:, np.newaxis],
#                                0)/ovlps

# results['du'] = (sums,maus,saus)

def white_noise(N: int) -> np.ndarray:
    X = np.random.normal(0, 1, N)
    return X / np.sqrt(np.mean(X**2))


def overlap_iteration(window_array: np.ndarray, overlap: float) -> float:
    N = len(window_array)
    x = white_noise(N + overlap)
    p = spk(window_array*x[:N])
    return (np.sqrt(np.nanmean(p)), kl_div(as_pdf(p),
                                           as_pdf(spk(window_array*x[-N:]))))


def my_summary(x: np.ndarray) -> tuple:
    if len(x):
        return (len(x), np.nanmean(x), np.nanstd(x))
    return (np.nan, np.nan, np.nan)


def my_summarize(X: np.ndarray, a: np.ndarray, ix: np.ndarray) -> np.ndarray:
    return np.array([(i, *my_summary(np.compress(a == i, X))) for i in ix],
                    dtype=[('index', np.int32),
                           ('length', np.uint32),
                           ('mean kl', np.float32),
                           ('kl sd', np.float32)])


Oh = np.repeat(np.arange(40, 0, -1)/40, 250)
fft_sizes = 2**(np.arange(7)+6)
KL = np.zeros((len(Oh), len(fft_sizes)))
S = KL.copy()
KLsums = dict()
Ssums = dict()
BothSums = dict()

for j, n in enumerate(fft_sizes):
    W = hamming(n)
    a = np.int32(Oh*n)
    for i, o in enumerate(a):
        S[i, j], KL[i, j] = overlap_iteration(W, o)
    ix = np.array([x for x in np.arange(n) if x in a])
    kl = KL[:, j]
    KLsums[n] = my_summarize(kl, a, ix)
    s = np.sqrt(S[:, j])
    Ssums[n] = my_summarize(s, a, ix)
    BothSums[n] = my_summarize(s*kl, a, ix)

for n, X in sorted(KLsums.items()):
    plt.errorbar(100*(1-X['index']/n),
                 X['mean kl'],
                 X['kl sd']/np.sqrt(X['length']),
                 label=n)

plt.legend(loc='best')
plt.xlabel('Percent Overlap')
plt.ylabel('Mean KL divergence')
plt.show()
