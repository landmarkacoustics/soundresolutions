# Copyright (C) 2018 by Landmark Acoustics LLC


def bandslice(lo:float, hi: float, fft_size: int, samplerate: float)->slice:
    a = int(lo*fft_size/samplerate)
    return slice(a,a+int(np.ceil(hi*fft_size/samplerate)))

from pysndfile import PySndfile as Sound
tres = dict()
for k in ['a','b','c','d']:
    tres[k] = Sound("/home/ben/Documents/Landmark Acoustics/Technology Development/Optimal Overlap/027 20 May " + k + ".aiff").read_frames()

tres['whole'] = Sound("/home/ben/Documents/Sounds/027 20 May.wav").read_frames()

slides = np.zeros((50,len(fft_sizes),len(tres)))
SLECHS = dict()
for fft_size in fft_sizes:
    SLECHS[fft_size] = (SR.SpectrogramMachine(signal.hamming(fft_size)), WDC(fft_size,50,23))

for j, (k,x) in enumerate(sorted(tres.items())):
    print(k)
    for i, (fft_size, (M,C)) in enumerate(sorted(SLECHS.items())):
        sly = bandslice(1000,8000,fft_size,44100)
        S = M(x,1)[:,sly]
        rms = np.sqrt(np.mean(S))
        C.update(S)
        slides[:,i,j] = np.sqrt(C.weighted_divergences()**2 / fft_size) / rms
        print(fft_size)

for j, (k,x) in enumerate(sorted(tres.items())):
    duration = len(tres[k])/44100
    plt.figure()
    plt.suptitle(k)
    for i, (fft_size, (M,C)) in enumerate(sorted(SLECHS.items())):
        ix = C.steps()[slides[:,i,j].argmax()]
        sly = bandslice(0,10000,fft_size,44100)
        plt.subplot(2,3,i+1)
        S = -10*np.log10(M(tres[k],ix)[:,sly])
        S[S>30.0] = 30.0
        plt.imshow(S.T, origin='lower',aspect='auto',interpolation='none',extent=[0,duration,0,10000]);
        plt.gray()
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.text(0,10000,'FFT size = ' + str(fft_size),horizontalalignment='left',verticalalignment='top')
        plt.text(duration,10000,'Overlap = {}%\nMWIED = {:>3}'.format(int(100*(1-ix/fft_size)),round(slides[:,i,j].max(),3)),horizontalalignment='right',verticalalignment='top')

plt.show()
