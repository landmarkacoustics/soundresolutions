np.random.seed(42)
divs = np.zeros([len(phis), len(fft_sizes), 2])
half = int((N * duration) // 2)
for i, phi in enumerate(phis):
    ampl = signal.sawtooth(2*np.pi*phi*time)
    for j, K in enumerate(fft_sizes):
        win = signal.gaussian(K, K/5)
        spek = sr.real_spk(win * ampl[half - K//2: half + K//2])
        divs[i,j,0] = euk(spek)
        divs[i,j,1] = np.corrcoef(spek[1:],spek[:-1])[0,1]

