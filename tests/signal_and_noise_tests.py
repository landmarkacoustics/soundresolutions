class Noisemaker:
    def __init__(self, sample_rate:float, duration:int, signal_function)->None:
        self._Hz = sample_rate
        self._times = np.arange(int(sample_rate * duration)) / sample_rate
        self._signal = signal_function(self._times)
    def white_noise(self)->np.ndarray:
        wn = np.random.normal(0,1,len(times))
        return wn/max(wn)
    def noise_and_sound(snr:float=1)->np.ndarray:
        k = snr / (1 + snr)
        return (1 - k) * self.blanco() + k * self._signal

sample_rate = 44100
N = sample_rate // 5
duration = N / sample_rate

noisemaker = Noisemaker(sample_rate, duration, lambda t: signal.sawtoooth(2*np.pi*phi*t))

n_steps = 50
n_repeats = 7

tied_computer = TiedComputer(n_steps, n_repeats)

noise_dict = dict(tenth=0.1, fifth=0.2, half=0.5, one=1, two=2, five=5, ten=10)

harms_and_noise = dict()

        resolution_results = dict()
        for k, snr in self._noise_dict.items():
            wave = noisemaker.noise_and_sound(snr)


class ResolutionContext:
    def __init__(self, fft_sizes:np.ndarray=2**np.arange(6,12), n_steps:int=50, n_starts:int=7)->None:
        self._computers = dict()
        for fft_size in fft_sizes:
            self._computers[fft_size] = TiedComputer(fft_size, n_steps, n_starts)
        self._values = np.tile(0, [n_steps, len(fft_sizes), 3])
    def __call__(self, wave, window_name:str='blackman'):
        for j, (fft_size, computer) in enumerate(self._computers.items()):
            spg = spectrogram(wave, signal.get_window(window_name, fft_size))
            computer.update(spg)
            self._values[:,j,:] = computer.values()
    def alienations(self)->np.ndarray:
        return self._values[:,:,0]
    def distances(self)->np.ndarray:
        return self._values[:,:,1]
    def TIEDs(self)->np.ndarray:
        return self._values[:,:,2]
    def values(self)->np.ndarray:
        return self._values
    

class SpectralContext:
    def __init__(self, wave, tied_computer:TiedComputer, window:np.ndarray)->None:
        self._fft_size = len(window)
        self._spg = spectrogram(wave, self._window)
        self._R, self._D, self._RD = tied_computer(spg)
    def steps(n:int)->np.ndarray:
        return np.linspace(1,self._fft_size,n,False)
    def alienations(self)->np.ndarray:
        return self._R
    def distances(self)->np.ndarray:
        return self._D
    def TIEDs(self)->np.ndarray:
        return self._RD
        
for k,v in harms_and_noise.items():
	plt.figure()
	for j,fft_size in enumerate(fft_sizes):
		steps =np.linspace(1,fft_size,v['RD'].shape[0],False)/44100
		plt.plot(steps, v['RD'][:,j]/7/44100, label=fft_size);
	plt.title(k);plt.xscale('log');plt.legend(loc='best');
