# Copyright (C) 2018 by Landmark Acoustics LLC
r"""Classes for making spectra and spectrograms."""

import numpy as np
from scipy.fftpack import fft, ifft


def dft_frequencies(Hz: float, fft_size: int) -> np.ndarray:
    r"""Calculates the frequency values of a DFT

    These are the lower bounds of the bins because that's easier.

    Parameters
    ----------
    Hz : float
        The sample rate of the sound.
    fft_size : int
        The length of the DFT.

    Returns
    -------
    out : np.ndarray
        The lower bounds of each frequency bin, in [0, Nyquist]

    """

    return np.linspace(0, Hz/2, 1+fft_size//2)


def dft_quefrencies(Hz: float, fft_size: int) -> np.ndarray:
    r"""Calculates the quefrency values of a cepstrum

    These are the centers of the bins because that's easier.

    Parameters
    ----------
    Hz : float
        The sample rate of the sound.
    fft_size : int
        The length of the DFT.

    Returns
    -------
    out : np.ndarray
        The center quefrencies, in the same units as `Hz`.

    """

    return Hz / np.arange(1, 1+fft_size//2)


def full_cpk(spk: np.ndarray) -> np.ndarray:
    r"""Finds periodicity in a log-transformed power spectrum.

    Good for extracting formants and fundamental frequency information.

    Parameters
    ----------
    spk : np.ndarray
        A power spectrum that has been log-transformed.

        You will get really weird results if you try to pass a power
        spectrum that hasn't been log-transformed!

    Returns
    -------
    out : np.ndarray
        A complex-valued DFT that includes both positive and negative
        quefrency information.

    See Also
    --------
    scipy.fftpack.ifft : inverse Fourier transform

    Examples
    --------
    import numpy as np
    from scipy import signal
    from matplotlib import pyplot as plt
    from soundresolutions import full_cpk, Sawtooths, SpectrumMachine

    duration = 0.1
    sample_rate = 44100.0
    fzeros = np.linspace(100,600,6)
    carey = Sawtooths(dict(fundamental = fzeros),
                           dict(duration = duration,
                                Hz=sample_rate,
                                decay_eps=1e-15,
                                decay_degree=6))
    fft_size = 1024
    M = SpectrumMachine(fft_size)
    W = signal.hamming(fft_size)
    x = np.zeros(fft_size)
    y = np.zeros(fft_size//2)
    z = y.copy()
    q = sample_rate / np.arange(1,len(z)+1)
    for D, X in carey:
        n = len(X)
        if n < fft_size:
            offset = (fft_size - n) // 2
            x[offset : offset + n] = X
        else:
            offset = (n - fft_size) // 2
            x = X[offset : offset + fft_size]

        y = M(W * x)
        z = full_cpk(10*np.log10(y))[:len(y)]
        plt.plot(q,20*np.log10(abs(z)),label=D['fundamental'])
        plt.xlim([q[-1],2000.0])

    plt.legend(loc='best',title='F0')
    plt.show()
    """

    return ifft(np.concatenate([spk, spk[-1:None:-1]]))


def lifter(spk: np.ndarray,
           mask: np.ndarray = None,
           cutoff: np.ndarray = None) -> np.ndarray:
    r"""Finds filter parts of the source-filter process that made spk.

    The source of the process cannot have a fundamental frequency
    greater than the frequency that corresponds to `cutoff`.
    Alternatively, you can specify the precise quefrency bins that
    make up the lifter using the `mask` array.

    Parameters
    ----------
    spk : np.ndarray
        A real-valued, 1-D, log-transformed power-spectrum.
    mask : np.ndarray, optional
        A boolean mask that removes the quefrencies of the source.
    cutoff : np.ndarray, optional
        The index of the greatest frequency that the source could have.

    Returns
    -------
    out : np.ndarray
        A real-valued, 1-D, low-pass smooth of `spk`.

    See Also
    --------
    scipy.fftpack.fft : forward FFT
    full_cpk : computes the cepstrum.

    Examples
    --------
    TBD

    """

    half_fft = len(spk)

    if mask is None:
        if cutoff is None or cutoff > half_fft:
            e = ValueError('you must pass a mask or a valid cutoff')
            raise e

        mask = np.repeat(1, 2*half_fft)
        mask[cutoff:-cutoff] = 0

    return fft(full_cpk(spk)*mask).real[:half_fft]


def real_spk(x: np.ndarray) -> np.ndarray:
    """Finds the power spectrum of real-valued inputs.

    Parameters
    ----------
    x : np.ndarray
        A one-dimensional array that should be a time series like sound.

    Returns
    -------
    out : np.ndarray
        The power spectrum of `x` :math:`|F(x)|^2`

    See Also
    --------
    numpy.fft.rfft : Computes the complex-valued spectrum of its real input.

    Examples
    --------
    The root-mean-square of the input is equal to the root mean of the
    power spectrum:

    >>> np.sqrt((np.arange(128)**2).mean())
    73.467679968813499
    >>> np.sqrt(real_spk(np.arange(128)).mean())
    73.467679968813499

    """

    return np.square(np.abs(np.fft.rfft(x)))/len(x)


class SpectrumMachine:
    r"""Finds the power spectrum of one-dimensional inputs.

    Parameters
    ----------
    fft_size : int
        The length of the input to the DFT

    See Also
    --------
    scipy.fftpack.fft : discrete Fourier transform

    Examples
    --------
    TBD

    """

    def __init__(self, fft_size: int):
        self._fft_size = fft_size
        self._half_fft = fft_size // 2

        self._slice = slice(1, self._half_fft)
        self._cf_idx = [0, self._half_fft]

        self._complex = np.zeros(self.spectrum_size(), dtype=complex)
        self._real = np.zeros(self.spectrum_size())

    def spectrum_size(self) -> int:
        return self._fft_size + 1

    def __call__(self, X: np.ndarray) -> np.ndarray:
        r"""Calculates the DFT of the input.

        Note that the DFT doesn't actually care what the sample rate of
        `X` is, so that is not part of either the initialization or the
        actual calling of the function. You can get resolutions out from
        other instance methods, though.

        Parameters
        ----------
        X : np.ndarray
            A one-dimensional, real-valued input that has the same
            length as the FFT size passed in at the instance's creation.

        Returns
        -------
        out : np.ndarray
            A 1-D, real-valued array of energies at each frequency.

        See Also
        --------
        scipy.fftpack.fft : discrete Fourier transform
        real_spk : a lower-overhead, but less efficient, approach

        Examples
        --------
        TBD

        """

        if len(X) != self._fft_size:
            raise ValueError("wrong size input to SpectrumMachine")

        self._complex[:] = np.fft.rfft(X)

        self._real[:] = np.square(np.abs(self._complex)) / self._fft_size

        return self._real

    def fft_size(self) -> np.float:
        r"""The size of the DFT"

        Returns
        -------
        fft_size : int

        Examples
        --------
        >>> S = SpectrumMachine(16)
        >>> S.fft_size()
        16

        """

        return self._fft_size

    def resolution(self, sample_rate: float) -> float:
        r"""The frequency resolution of the DFT at the sample rate.

        Parameters
        ----------
        sample_rate : float
            The number of samples per second of an input sound.

        Returns
        -------
        out : float
            The smallest frequency that the DFT can resolve.

        Examples
        --------
        >>> SpectrumMachine(16).resolution(44100)
        2756.25

        """

        return sample_rate / self._fft_size

    def frequencies(self, sample_rate: float) -> np.array:
        r"""The lower bounds of the frequency bins of the DFT output.

        A DFT of length *n* will find the energy in *n/2* frequency
        ranges, or 'bins'. This function returns the lower boundary
        of each of those bins. The upper boundary of the series is the
        Nyquist frequency, or `sample_rate` / 2.

        Parameters
        ----------
        sample_rate : float
            The number of samples per second of an input sound.

        Returns
        -------
        out : np.ndarray
            An array of frequency values.

        Examples
        --------
        >>> SpectrumMachine(8).frequencies(11025)
        array([    0.   ,  1378.125,  2756.25 ,  4134.375, 0])

        """

        return dft_frequencies(sample_rate, self._fft_size)


class SpectrogramMachine:
    r"""Contains buffers for reasonably rapid spectrogram computations.

    The idea of this class is to step through the input array and
    compute a spectrogram that has some zero-padding on the edges

    Parameters
    ----------
    window_function : np.ndarray
        The weights to use when multiplying the inputs.

        Note that we also get the FFT size from this array's length.

    See Also
    --------
    signal.get_window : A good way to generate `window_function` arrays.

    Examples
    --------
    TBA

    """

    def __init__(self, window_function: np.ndarray) -> None:
        self._spk = SpectrumMachine(len(window_function))
        self._W = window_function.copy()
        self._buffer = np.zeros(len(self._W))

    def __call__(self,
                 X: np.ndarray,
                 step: int = None,
                 Y: np.ndarray = None) -> np.ndarray:
        r"""Computes the spectrogram of the input.

        Parameters
        ----------
        X : np.ndarray
            The input waveform. A 1-D, real-valued time series.
        step : int, optional
            The number of samples between spectra.
        Y : np.ndarray, optional
            An optional array to hold the output, if you want to.

        Returns
        -------
        Y : np.ndarray
            A real-valued power spectrogram of `X`.

        """
        should_return = True

        if Y is None:
            if step is None:
                step = self._spk._half_fft
            Y = np.zeros([int(np.ceil(len(X)/step)), 1 + self._spk._half_fft])
            should_return = True

        xlen = X.shape[0]
        ylen = Y.shape[0]

        for i, x in enumerate(self.step_locations(xlen, ylen)):
            start, lpads = self._left_indices(x)
            stop, rpads = self._right_indices(x, xlen)
            self._buffer[:lpads] = 0.0
            self._buffer[lpads:rpads] = X[start:stop]
            self._buffer[rpads:] = 0.0
            Y[i, :] = self._spk(self._buffer * self._W)

        return Y if should_return else None

    def _left_indices(self, center: int) -> tuple:
        leftmost = center - self._spk._half_fft
        if leftmost < 0:
            return (0, -leftmost)
        return (leftmost, 0)

    def _right_indices(self, center: int, N: int) -> tuple:
        rightmost = center + self._spk._half_fft
        if rightmost > N:
            return (N, self._spk.fft_size() - (rightmost - N))
        else:
            return (rightmost, self._spk.fft_size())

    def step_locations(self, xlen: int, ylen: int) -> np.ndarray:
        r"""Finds the indices of the windows that make a spectrogram.

        Parameters
        ----------
        xlen : int
            The length of the input time series
        ylen : int
            The length (num. of columns) of the output spectrogram.

        Returns
        -------
        out : np.ndarray
            An array of indices into whatever has `xlen` items.

        """

        return np.array(np.around(np.linspace(0, xlen, ylen, False), 0),
                        dtype=int)
