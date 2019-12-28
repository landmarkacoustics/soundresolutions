# Copyright (C) 2018 by Landmark Acoustics LLC

import numpy as np
from scipy import signal
from scipy import fftpack
from scipy import stats
from matplotlib import pyplot as plt
import soundresolutions as sr

from collections import OrderedDict

class Sampler:
    def __init__(self, sample_rate: float) -> None:
        self._N = sample_rate

    @property
    def sample_rate(self):
        r'''The rate at which sounds will be sampled'''
        return self._N
    
    def times(self, length: int) -> np.ndarray:
        r'''A span of `length` times separated by 1/`sample_rate`'''
        k = 0.5 * length / self.sample_rate
        return np.arange(-k, k, 1 / self.sample_rate)

    def length(self, duration: float) -> int:
        r'''The number of samples required for a window with the given duration.'''
        return int(np.ceil(duration * self.sample_rate))


class Emitter:
    def __init__(self, sample_rate: float, smallest_size: int, largest_size: int, count: int) -> None:
        self._N = sample_rate
        self._frequencies = self.optimal_frequencies(2**np.linspace(np.log2(largest_size), np.log2(smallest_size), count))
        
    @property
    def sample_rate(self):
        r'''The sample rate of the sounds in this context'''
        return self._N

    @property
    def frequencies(self):
        r'''The span of frequencies the emitter traverses.'''
        return self._frequencies

    def iterate_for_one_time_range(self, t: np.ndarray):
        for phi in self.frequencies:
            yield self(phi, t)
    
#    def optimal_frequencies(self, wavelengths_in_samples):
#        return self.sample_rate / wavelengths_in_samples

#    def call(self, frequency: float, instants: np.ndarray) -> np.ndarray:
#        return np.sin(2*np.pi*instants*frequency)
    
class FMWhistle(Emitter):
    def optimal_frequencies(self, window_lengths):
        return (self.sample_rate / window_lengths)**2
    
    def __call__(self, frequency: float, instants: np.ndarray) -> np.ndarray:
        return np.sin(-2*np.pi*instants*(0.5*frequency*instants + self.sample_rate/4))

class Sawtooth(Emitter):
    def optimal_frequencies(self, window_lengths):
        return 2 * self.sample_rate / window_lengths

    def __call__(self, frequency: float, instants: np.ndarray) -> np.ndarray:
        return signal.sawtooth(2*np.pi*frequency*instants)

class PulseTrain(Emitter):
    def optimal_frequencies(self, window_lengths):
        return 2 * self.sample_rate / window_lengths

    def __call__(self, frequency: float, instants: np.ndarray) -> np.ndarray:
        n = len(instants)
        train = np.zeros(n, dtype = float)
        pulse = np.remainder(np.arange(-n//2, n//2),
                             int(np.around(self.sample_rate/frequency)))
        train[pulse < 1] = 1.0
        return train / np.sqrt(np.nanmean(np.square(train)))

class WhiteNoise(Emitter):
    def optimal_frequencies(self, window_lengths):
        return self.sample_rate / window_lengths
    def __call__(self, frequency: float, instants: np.ndarray) -> np.ndarray:
        return np.random.normal(0,1,len(instants))

class Context:
    def __init__(self, fft_sizes: np.ndarray, measures: OrderedDict, sample_rate: float = 44100.0) -> None:
        self._sizes = fft_sizes
        self._measures = measures
        self._sampler = Sampler(sample_rate)

    @property
    def fft_sizes(self):
        return self._sizes

    @property
    def measures(self):
        return self._measures

    def __call__(self, emitter: Emitter, data: np.ndarray = None) -> np.ndarray:
        proper_shape = (emitter.frequencies.size, len(self.fft_sizes), len(self.measures))
        if data is None or data.shape != proper_shape:
            data = np.zeros(proper_shape)

        for j, K in enumerate(self.fft_sizes):
            t = self._sampler.times(K)
            W = signal.gaussian(K,K/5)
            for i, a in enumerate(emitter.iterate_for_one_time_range(t)):
                spek = sr.real_spk(W*a)
                for k, f in enumerate((self.measures.values())):
                    data[i,j,k] = f(spek)

        return data

def H(x):
    f = x > 0.0
    p = x[f] / np.nansum(x[f])
    return -np.nansum(p*np.log(p))

def rms(x):
    return np.sqrt(np.nanmean(x))

def al(x):
    return np.nansum(np.abs(np.diff(x)))

def wiener(x):
    return stats.gmean(x) / np.nanmean(x)

# funcs = OrderedDict([('entropy', H),
#                      ('energy', rms),
#                      ('arc length', al),
#                      ('variance', lambda x : np.std(x,ddof=1))])

#funcs = OrderedDict([('population', np.var),
#                     ('sample', lambda x : np.var(x,ddof=1))])

funcs = OrderedDict([
    ('variance', np.var),
    ('wiener', wiener),
    ('arclength', al),
    ('entropy', H),
    ('energy', rms)
])


N = 44100

fft_low = 6
fft_top = 12
n_frequencies = 1000

fft_sizes = 2**np.arange(fft_low, fft_top)

contrary = Context(fft_sizes, funcs, N)

emitters = OrderedDict()
for n, E in [('whistle',FMWhistle), ('hum',Sawtooth), ('clicks', PulseTrain), ('shhh', WhiteNoise)]:
    emitters[n] = E(N, 2**(fft_low-1), 2**fft_top, n_frequencies)

holds = OrderedDict([(n,contrary(E)) for n,E in emitters.items()])

nrows = int(np.sqrt(len(emitters)))
ncols = int(np.ceil(len(emitters) / nrows))

for k, funcname in enumerate(funcs.keys()):
    plt.figure()
    for i, (n,e) in enumerate(emitters.items()):
        plt.subplot(nrows, ncols, i+1) and None
        tmp = holds[n][:,:,k]
        p = e.optimal_frequencies(fft_sizes)
        for j, K in enumerate(fft_sizes):
            plt.plot(e.frequencies, tmp[:,j], label = K) and None
            
        plt.vlines(p, np.nanmin(tmp), np.nanmax(tmp)) and None
        plt.xscale('log') and None
        plt.xlabel('Frequency')
        plt.ylabel(funcname) and None
        plt.legend(loc='best') and None
        plt.title(n) and None

plt.show()

freaks = np.array(np.around(np.exp(np.linspace(np.log(20), np.log(2e6), n_frequencies))), dtype=int)
for k, L in [('FM', lambda m : N / np.sqrt(m)),
             ('Sawtooth', lambda phi : 2*N/phi),
             ('Pulse Train', lambda r : N/r)]:
    plt.plot(freaks, [fftpack.next_fast_len(int(np.around(L(x)))) for x in freaks], label = k)

plt.hlines(fft_sizes, freaks[0], freaks[-1])
plt.xscale('log')
plt.xlabel('Value of Parameter')
plt.yscale('log')
plt.ylabel('Predicted Best FFT size')
plt.legend(loc='best')
plt.show()

whoa = OrderedDict()
for K in fft_sizes:
    whoa[K] = sr.WDC(K,
                     ('gaussian',K/5),
                     wiv
                     )

overlaps = np.linspace(0,1,25,False)
batch = sr.BatchWDC(whoa, 23, overlaps)

tempo = Sampler(N).times(5000)
results = OrderedDict()

lo_power = 4
hi_power = 20
detailed_freqs = 2**np.linspace(lo_power,hi_power,hi_power-lo_power,dtype=int)

for k, e in emitters.items():
    results[k] = np.zeros([detailed_freqs, 2])
    for i, phi in enumerate(e.frequencies[::delta]):
        batch(e(phi, tempo))
        results[k][i,:] = batch.best_sums()


from pysndfile import PySndfile as Sound
spek = Sound('/home/ben/Documents/Sounds/soundresolutions/spectrogram.wav').read_frames()

batch(spek)

spectrogram_results = batch.sums()

phonemes = ['s','p','e','k','t','u','g','r','a','m']
byphoneme = OrderedDict()
min_length = 2*max(fft_sizes)
for i, p in enumerate(phonemes):
    tmp = Sound('/home/ben/Documents/Sounds/soundresolutions/spec' + str(i) + '.wav').read_frames()
    if len(tmp) < min_length:
        tmp = np.concatenate((tmp, np.random.normal(0,1/100,min_length-len(tmp))))
    
    batch(tmp)
    byphoneme[p] = batch.sums()
    print(p)

tres = OrderedDict()
for i, p in enumerate(['a','b','c','d']):
    tmp = Sound('/home/ben/Documents/Sounds/soundresolutions/027 20 May ' + str(p) + '.wav').read_frames()
    if len(tmp) < min_length:
        tmp = np.concatenate((tmp, np.random.normal(0,1/100,min_length-len(tmp))))
    
    batch(tmp)
    tres[p] = batch.sums()
    print(p)

duration = 1
time = np.arange(0,duration,1/N)
mmin = 1
mmax = 1e6
accel = (mmax - mmin) / duration / 2
ampl = np.sin(-2*np.pi*time*(accel*time**2/3 + mmin * time / 2))
#plt.plot(time[1:], N*np.unwrap(np.diff(np.angle(signal.hilbert(ampl+np.random.normal(0,1/20,len(time))))))/np.pi/2)
#plt.plot(time,accel*time**2)
#plt.ylim(0,N/2)
#plt.show()


partials_around = lambda tees : np.array([1,*np.around(N*tees),-1],dtype=int)

optimal_whistle_times = duration * (N**2 - mmin*fft_sizes**2)/((mmax-mmin)*fft_sizes**2)
whistles_indices = partials_around(optimal_whistle_times[::-1])
nwhistles = len(whistles_indices)
part_whistles = np.zeros((nwhistles, len(overlaps), len(fft_sizes)))

quiet_noise = lambda : np.random.normal(0,1/20,len(time))

for i, ix in enumerate(whistles_indices):
    batch(quiet_noise() + kernel(time,time[ix],duration/20)*ampl)
    part_whistles[i,:,:] = batch.sums() * fft_sizes / N

best = np.max(part_whistles)

for j, K in enumerate(fft_sizes):
    plt.subplot(2,3,j+1)
    plt.title(f'FFT size = {K}')
    for i, a in enumerate(2*accel*time[whistles_indices]):
        plt.plot(overlaps, part_whistles[i,:,j], label = f'{a:2.1g}')

    plt.ylim(0,best)
    plt.xlabel('Overap Between Windows (%)')
    plt.ylabel(f'Weighted Variance ($V^2$)')
    plt.legend(loc='best')

plt.show()

fmin = 40
fmax = 1600
sawslope = (fmax - fmin) / duration
saws = signal.sawtooth(-2*np.pi*time*(sawslope*time/2+fmin))

optimal_saw_times = duration * (2*N - fmin*fft_sizes)/((fmax-fmin)*fft_sizes)
saws_indices = partials_around(optimal_saw_times[::-1])
nsaws = len(saws_indices)
part_saws = np.zeros((nsaws, len(overlaps), len(fft_sizes)))

for i, ix in enumerate(saws_indices):
    batch(quiet_noise() + kernel(time,time[ix],duration/20)*ampl)
    part_saws[i,:,:] = batch.sums() * fft_sizes / N

best = np.max(part_saws)

for j, K in enumerate(fft_sizes):
    plt.subplot(2,3,j+1)
    plt.title(f'FFT size = {K}')
    for i, a in enumerate((fmax-fmin)*time[saws_indices]/duration+fmin ):
        plt.plot(overlaps, part_saws[i,:,j], label = f'{int(a)}')

    plt.ylim(0,best)
    plt.xlabel('Overap Between Windows (%)')
    plt.ylabel(f'Weighted Variance ($V^2$)')
    plt.legend(loc='best')

plt.show()

