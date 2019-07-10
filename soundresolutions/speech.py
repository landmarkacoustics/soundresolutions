# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from pysndfile import PySoundFile as Sound
from collections import OrderedDict

# import soundresolutions as SR


# Hz = 44100
# duration = 0.1
# nmin = len(SR.times(duration,Hz))

# fft_sizes = 2**np.arange(6,12)

# spekpath = '/home/ben/Documents/Sounds/soundresolutions'
# speknames = ['s','p','e','k','t','ruh','g','r','a','m']
# speks = OrderedDict()
# for i, k in enumerate(speknames):
#     speks[k] = Sound(spekpath + '/spec' + str(i) + '.wav').read_frames()
# whole_spek = Sound(spekpath + '/spectrogram.wav').read_frames()

# nsteps = 50
# nstarts = 23
# SPECHS = dict()
# for fft_size in fft_sizes:
#     SPECHS[fft_size] = (SR.SpectrogramMachine(signal.hamming(fft_size)),
#                         SR.WeightedDivergenceComputer(fft_size,nsteps,nstarts))

# spides = np.zeros((nsteps, len(fft_sizes), len(speks)))

# nw = SR.proportions_from_snr(20)

# for j, k in enumerate(speknames):
#     print('\n/' + k + '/')
    
#     x = speks[k]
#     n = len(x)
#     if n < nmin:
#         tmp = (nmin-n)//2
#         sly = slice(tmp,tmp+n)
#         n = nmin
#     else:
#         sly = slice(0,n)eighted_divergences()**2/fft_size) / rms
#         print(fft_size)

# for j, k in enumerate(speknames):
#     plt.figure()
#     plt.suptitle(k)
#     for i, (fft_size, (Spg, Wdc)) in enumerate(sorted(HECHS.items())):
#         plt.plot(100*(1-Wdc.steps()/fft_size), spides[:,i,j], label = fft_size)
#         plt.legend(loc='best',title="FFT size")

# plt.show()

# for j, k in enumerate(speknames):
#     x = speks[k]
#     duration = len(speks[k])/44100
#     plt.figure()
#     plt.suptitle('/'+k+'/')
#     for i, (fft_size, (M,C)) in enumerate(sorted(SPECHS.items())):
#         ix = C.steps()[spides[:,i,j].argmax()]
#         sly = bandslice(0,10000,fft_size,44100)
#         plt.subplot(2,3,i+1)
#         S = -10*np.log10(M(speks[k],ix)[:,sly])
#         S[S>30.0] = 30.0
#         plt.imshow(S.T, origin='lower',aspect='auto',interpolation='none',extent=[0,duration,0,10000]);
#         plt.gray()
#         plt.xlabel('Time (s)')
#         plt.ylabel('Frequency (Hz)')
#         plt.text(0,10000,'FFT size = ' + str(fft_size),horizontalalignment='left',verticalalignment='top')
#         plt.text(duration,10000,'Overlap = {}%\nMWIED = {:>3}'.format(int(100*(1-ix/fft_size)),round(spides[:,i,j].max(),3)),horizontalalignment='right',verticalalignment='top')

# plt.show()

# a = whole_spek
# wholespide = np.zeros((nsteps, len(fft_sizes)))

# rms = np.sqrt(np.mean(a**2))

# for j, (fft_size, (Spg, Wdc)) in enumerate(sorted(SPECHS.items())):
#     sly = bandslice(0,8000,fft_size,Hz)
#     Wdc.update(Spg(a,1)[:,sly],
#                SR.euclidean_distance,
#                SR.coefficient_of_alienation)
#     wholespide[:,j] = np.sqrt(Wdc.weighted_divergences()**2/fft_size) / rms
#     print(fft_size)

# duration = len(whole_spek)/44100
# for j, (fft_size, (M,C)) in enumerate(sorted(SPECHS.items())):
#     ix = C.steps()[wholespide[:,j].argmax()]
#     plt.subplot(2,3,j+1)
#     plt.imshow(10*np.log10(M(whole_spek,ix)).T,
#                origin='lower',
#                aspect='auto',
#                extent = [0,duration,0,Hz/2],
#                interpolation='none',
#                cmap = 'gray_r')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Frequency (Hz)')
#     plt.ylim([0,8000])
#     plt.title('FFT size = ' + str(fft_size))

# plt.show()

# for j, (fft_size, (Spg,Wdc)) in enumerate(sorted(SPECHS.items())):
#     plt.plot(100*(1-Wdc.steps()/fft_size), wholespide[:,j], label =fft_size)

# plt.legend(loc='best')
# plt.show()

#proabably better

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from pysndfile import PySndfile as Sound
import soundresolutions as sres

n_offsets = 50

phonemes = ['s','p','e','k','t','er','g','r','a','m']
path = '/home/ben/Documents/Sounds/soundresolutions/'
base = 'spec'
suff = '.wav'
for i,k in enumerate(phonemes):
    fn = path + base + str(i) + suff
    specparts[k] = Sound(fn).read_frames()

specparts['full'] = Sound(path + 'spectrogram' + suff).read_frames()

fft_sizes = 2**np.arange(6,12)
for fft_size in fft_sizes:
    machines[fft_size] = sres.SpectrogramMachine(signal.hamming(fft_size))

specmieds = OrderedDict()

print('\n')
dims = (len(fft_sizes), n_offsets)
O = np.tile(0,dims)
for i, fft_size in enumerate(fft_sizes):
    O[i,:] = np.linspace(1,fft_size,n_offsets,False,dtype=int)

for phone, wave in specparts.items():
    specmieds[phone] = np.tile(np.nan,dims)
    B = np.tile(0.0, [len(wave),n_offsets])
    print(phone)
    for i, (fft_size, SM) in enumerate(machines.items()):
        specmieds[phone][i,:] = mieds(SM(wave,1), O[i,:], B)
        print(fft_size)

for j, (phone, M) in enumerate(specmieds.items()):
    plt.subplot(3,4,j+1) and None
    rms = np.sqrt(np.nanmean(np.square(specparts[phone])))
    for i, fft_size in enumerate(fft_sizes):
        plt.plot(100*(1-O[i,:]/fft_size), M[i,:]/rms) and None
    
    plt.title(phone) and None

plt.subplot(3,4,12) and None
for i, fft_size in enumerate(fft_sizes):
    plt.plot(0,0, label = fft_size) and None

plt.legend(loc='lower left',title='FFT size') and None
plt.show()
o
