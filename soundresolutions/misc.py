# Copyright (C) 2018 by Landmark Acoustics LLC

#### From the original whistles analysis ####
    

        
class WhistleContext:
    def __init__(self,
                 sample_rate: float,
                 duration: float,
                 bandwidths: np.ndarray,
                 snrs: np.ndarray) -> None:
        self._Hz = sample_rate
        self._Nyquist = 0.5 * self._Hz
        self._halffreq = 0.5 * self._Nyquist
        self._duration = duration
        self._halftime = 0.5 * duration
        self._fulltime = 0.95 * duration
        self._time = times(duration, sample_rate)
        self._bandwidths = bandwidths
        self._snrs = snrs
    def make_sound(self, bandwidth_fraction: float, SNR: float) -> np.ndarray:
        a, b = proportions_from_snr(SNR)
        return  (a * fm_pulse(self._time,
                              self._halftime,
                              self._halffreq,
                              self._fulltime,
                              self._Nyquist * bandwidth_fraction) +
                 b * white_noise(self._duration,
                                 self._Hz))
    def __iter__(self):
        for snr in self._snrs:
            for bandwidth in self._bandwidths:
                yield (snr, bandwidth, self.make_sound(bandwidth, snr))

        
fractions = [1/10,1/4,1/2,2/3,3/4]

fft_sizes = 2**np.arange(6,12)

whistles = np.zeros([len(fractions)**2, 3 + 2 * len(fft_sizes)])

whistles[:,0] = duration * np.array(len(fractions) * fractions)

whistles[:,1] = Ny * np.repeat(fractions, len(fractions))

whistles[:,2] = whistles[:,1] / whistles[:,0]

WF = Whistles(44100.0, 0.1)



i = 0

for snr in [19, 11, 7, 3, 2, 1]:
    for bandwidth in [.1, .3, .5, .7, .9]:
        s = WF(bandwidth, snr)
        for j, fft_size in enumerate(fft_sizes):
            results[j,:] = DCs[fft_size](s)
            
        yield bandwidth, snr, results
        
    print('slope: {:8}, Items[{}][{}] = {}'.format(whistles[i,2],
                                                   fft_sizes[3],
                                                   whistles[i,3+3],
                                                   whistles[i,3+3+len(fft_sizes)]))
    self._results = np.zeros([len(fft_sizes), 2])
    self._DCs = dict()
    for fft_size in fft_sizes:
        self._DCs[fft_size] = DSPContext(fft_size,
                                         'hamming',
                                         50,
                                         23,
                                         euclidean_distance,
                                         coefficient_of_alienation)

class WTH:
    def __init__(self, Hz: float, max_mult: int = 4, count: int = 10) -> None:
        self._Hz = Hz
        self._center_freq = Hz / 4
        self._maxmult=max_mult
        self._count = count
        self._bandwidths = np.linspace(0,self._Hz/2,count,False)
    def get_steps(self, fft_size: int) -> np.ndarray:
        return np.linspace(fft_size,
                           fft_size * self._maxmult,
                           self._count,
                           dtype=int)
    def generate(self, fft_size: int):
        for step in self.get_steps(fft_size):
            duration = step / self._Hz
            t = times(duration, self._Hz)
            halftime = 0.5 * duration
            for bandwidth in self._bandwidths:
                yield (duration, bandwidth, fm_pulse(t,
                                                     halftime,
                                                     self._center_freq,
                                                     duration,
                                                     bandwidth))

class SRSLY:
    def __init__(self,
                 sample_rate: float,
                 durations: np.ndarray,
                 bandwidths: np.ndarray,
                 SNR: float = 10):
        self._Hz = sample_rate
        self._midfreq = self._Hz / 4
        self._durations = durations
        self._bandwidths = bandwidths
        self._a, self._b = proportions_from_snr(SNR)
    def __len__(self):
        return len(self._durations) * len(self._bandwidths)
    def __iter__(self):
        for duration in self._durations:
            halftime = 0.5 * duration
            time = times(duration, self._Hz)
            for bandwidth in self._bandwidths:
                yield (duration,
                       bandwidth,
                       (self._a * fm_pulse(time,
                                           halftime,
                                           self._midfreq,
                                           duration,
                                           bandwidth)
                        + self._b * white_noise(len(time))))

class LengthByProportion:
    def __init__(self,
                 Hz: float,
                 whole_duration: float,
                 partial_durations: np.ndarray,
                 slope: float = None,
                 bandwidth: float = None,
                 SNR: float = 20):
        self._Hz = Hz
        self._midfreq = self._Hz / 4
        self._duration = whole_duration
        self._partial_durations = partial_durations
        self._time = times(self._duration, self._Hz)
        self._midtime = self._duration / 2
        self._weights = proportions_from_snr(SNR)
        self._bandwidths = np.zeros(len(self))
        if slope is not None:
            self._bandwidths[:] = [slope * duration for duration in self._partial_durations]
        else:
            if bandwidth is None:
                bandwidth = self._midfreq
                self._bandwidths[:] = bandwidth
    def __len__(self):
        return len(self._partial_durations)
    def __iter__(self):
        for duration,bandwidth in zip(self._partial_durations,self._bandwidths):
            yield (duration, bandwidth,
                   ( self._weights[0] * fm_pulse(self._time,
                                                 self._midtime,
                                                 self._midfreq,
                                                 duration,
                                                 bandwidth)
                     + self._weights[1] * white_noise(len(self._time))))


#### from the harmonic stack analysis ####


        
Hz = 44100
duration = 0.1
times = SR.times(duration, Hz)
fft_sizes = 2**np.arange(6,12)
SNR = 20
lowf0 = 20
hif0 = 300
nf0 = 20

fzeros = np.around(np.linspace(np.sqrt(lowf0),
                               np.sqrt(hif0),
                               nf0)**2,0)

nsteps = 50
nstarts = 23
HECHS = dict()
for fft_size in fft_sizes:
    HECHS[fft_size] = (SR.SpectrogramMachine(signal.hamming(fft_size)),
                       SR.WeightedDivergenceComputer(fft_size,nsteps,nstarts))

hides = np.zeros((nsteps, len(fft_sizes), len(fzeros)))

sublength = 0.08
degree = 8
am = np.exp(-SR.decay_rate(sublength,degree=degree)*(times - duration/2)**degree)
nw = SR.proportions_from_snr(SNR)

for j, phi in enumerate(fzeros):
    print(phi)
    a = (am * nw[0] * signal.sawtooth(phi * 2 * np.pi * times)
         + nw[1] * SR.white_noise(len(times)))
    rms = np.sqrt(np.mean(a**2))
    for i, (fft_size, (Spg, Wdc)) in enumerate(sorted(HECHS.items())):
        Wdc.update(Spg(a,1),
                   SR.euclidean_distance,
                   SR.coefficient_of_alienation)
        hides[:,i,j] = np.sqrt(Wdc.weighted_divergences()**2/fft_size) / rms
        print(fft_size)

# for j, phi in enumerate(fzeros):
#     plt.figure()
#     plt.suptitle("F0=" + str(phi))
#     for i, (fft_size, (Spg, Wdc)) in enumerate(sorted(HECHS.items())):
#         plt.plot(100*(1-Wdc.steps()/fft_size), hides[:,i,j], label = fft_size)
#         plt.legend(loc='best',title="FFT size")

# plt.show()

for i, fft_size in enumerate(fft_sizes):
    plt.plot(fzeros, np.max(hides[:,i,:],0), label = fft_size)

plt.legend(loc='best')
plt.xlabel('Fundamental Frequency of Sawtooth (Hz)')
plt.ylabel('MWIED (unitless)')
plt.show()
