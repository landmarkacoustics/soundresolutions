# Copyright (C) 2018 by Landmark Acoustics LLC

"""Find optimal resolutions for spectrograms that describe specific sounds.

This is a set of data-generating, algorithm, graphing, and test code to
go with Landmark Acoustic LLC's research on optimal spectrogram overlap.

Further docstrings will describe the organization of the package and its
components.

NumPy
=====
Provides
  1. An array object of arbitrary homogeneous items
  2. Fast mathematical operations over arrays
  3. Linear Algebra, Fourier Transforms, Random Number Generation
How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided
with the code, and a loose standing reference guide, available from
`the NumPy homepage <http://www.scipy.org>`_.
We recommend exploring the docstrings using
`IPython <http://ipython.scipy.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.
The docstring examples assume that `numpy` has been imported as `np`::
  >>> import numpy as np
Code snippets are indicated by three greater-than signs::
  >>> x = 42
  >>> x = x + 1
Use the built-in ``help`` function to view a function's docstring::
  >>> help(np.sort)
  ... # doctest: +SKIP
For some objects, ``np.info(obj)`` may provide additional help.  This is
particularly true if you see the line "Help on ufunc object:" at the top
of the help() page.  Ufuncs are implemented in C, not Python, for speed.
The native Python help() does not know how to view their help, but our
np.info() function does.
To search for documents containing a keyword, do::
  >>> np.lookfor('keyword')
  ... # doctest: +SKIP
General-purpose documents like a glossary and help on the basic concepts
of numpy are available under the ``doc`` sub-module::
  >>> from numpy import doc
  >>> help(doc)
  ... # doctest: +SKIP
Available subpackages
---------------------
doc
    Topical documentation on broadcasting, indexing, etc.
lib
    Basic functions used by several sub-packages.
random
    Core Random Tools
linalg
    Core Linear Algebra Tools
fft
    Core FFT routines
polynomial
    Polynomial tools
testing
    NumPy testing tools
f2py
    Fortran to Python Interface Generator.
distutils
    Enhancements to distutils with support for
    Fortran compilers support and more.
Utilities
---------
test
    Run numpy unittests
show_config
    Show numpy build configuration
dual
    Overwrite certain functions with high-performance Scipy tools
matlib
    Make everything matrices.
__version__
    NumPy version string
Viewing documentation using IPython
-----------------------------------
Start IPython with the NumPy profile (``ipython -p numpy``), which will
import `numpy` under the alias `np`.  Then, use the ``cpaste`` command to
paste examples into the shell.  To see which functions are available in
`numpy`, type ``np.<TAB>`` (where ``<TAB>`` refers to the TAB key), or use
``np.*cos*?<ENTER>`` (where ``<ENTER>`` refers to the ENTER key) to narrow
down the list.  To view the docstring for a function, use
``np.cos?<ENTER>`` (to view the docstring) and ``np.cos??<ENTER>`` (to view
the source code).

"""

__all__ = []

from .audio import *

__all__.extend(['am_tone', 'gauss_am', 'times', 'white_noise'])

from .autocorrelation import autocorrelation

__all__.extend(['autocorrelation'])

from .divergences import *

__all__ += ['euclidean_distance']

from .helpers import *
__all__ += ['proportions_from_snr','pSNR_from_dbSNR','rms','scale_to_unit_energy','whole_steps']

from .fourier import *
__all__ += ['dft_frequencies', 'dft_quefrencies', 'dft_reals', 'full_cpk', 'lifter', 'real_spk', 'SpectrumMachine', 'SpectrogramMachine']

from .pulse_train import *
__all__ += ['pulse_train']

from .sawtooths import Sawtooths

__all__.extend(['Sawtooths'])

from .scan_by_steps import scan_by_steps

__all__.extend(['scan_by_steps'])

from .stft import stft

__all__.extend(['stft'])

from .synthesizer import Synthesizer

__all__.extend(['Synthesizer'])

from .wdc import BatchWDC, WDC

__all__.extend('BatchWDC, WDC')

from .weighted_divergence_computer import WeightedDivergenceComputer
__all__ += ['WeightedDivergenceComputer']

from .weights import *

__all__ += ['coefficient_of_alienation']

from .wial import wial

__all__.extend('wial')

from .window_test import all_window_names, WindowTester

__all__.extend(['all_window_names', 'WindowTester'])

from .wiv import wiv

__all__.extend('wiv')

from .whistles import *

__all__.extend(['Whistles'])