# Copyright (C) 2018 by Landmark Acoustics LLC

r'''Find optimal resolutions for spectrograms that describe specific sounds.

This is a set of data-generating, algorithm, graphing, and test code to
go with Landmark Acoustic LLC's research on optimal spectrogram overlap.

Further docstrings will describe the organization of the package and its
components.

'''

from .audio import \
    am_tone, \
    gauss_am, \
    times, \
    white_noise, \
    TAU

from .autocorrelation import autocorrelation

from .divergences import euclidean_distance

from .helpers import \
    proportions_from_snr, \
    pSNR_from_dbSNR, \
    rms, \
    scale_to_unit_energy, \
    whole_steps

from .fourier import \
    dft_frequencies, \
    dft_quefrencies, \
    full_cpk, \
    lifter, \
    real_spk, \
    SpectrumMachine, \
    SpectrogramMachine

from .pulse_train import pulse_train

from .spaceranger import SpaceRanger

from .synthesizer import Synthesizer

from .sawtooths import Sawtooths

from .scan_by_steps import scan_by_steps

from .stft import stft

from .wdc import BatchWDC, WDC

from .weights import coefficient_of_alienation

from .weighted_divergence_computer import WeightedDivergenceComputer

from .wial import wial

from .window_test import all_window_names, WindowTester

from .wiv import wiv

from .whistles import Whistles


__all__ = [
    'am_tone',
    'gauss_am',
    'times',
    'white_noise',
    'TAU',
    'autocorrelation',
    'euclidean_distance',
    'proportions_from_snr',
    'pSNR_from_dbSNR',
    'rms',
    'scale_to_unit_energy',
    'whole_steps',
    'dft_frequencies',
    'dft_quefrencies',
    'full_cpk',
    'lifter',
    'real_spk',
    'SpectrumMachine',
    'SpectrogramMachine',
    'pulse_train',
    'Sawtooths',
    'scan_by_steps',
    'stft',
    'SpaceRanger',
    'Synthesizer',
    'BatchWDC',
    'WDC',
    'WeightedDivergenceComputer',
    'coefficient_of_alienation',
    'wial',
    'all_window_names',
    'WindowTester',
    'wiv',
    'Whistles',
]
