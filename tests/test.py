# Copyright (C) 2018 Landmark Acoustics, LLC

import unittest
import numpy as np
from . import audio

#helpers.py

from .helpers import *

class PropsFromSNRTestCase(unittest.TestCase):
    def test_p_from_snr(self):
        for i, snr in enumerate([.1,.2,.5,1,2,5,10]):
            with self.subTest(i=i):
                (k, notk) = proportions_from_snr(snr)
                self.assertAlmostEqual(k, snr / (1+snr))
                self.assertAlmostEqual(notk, 1 - k)

class WholeStepsTestCase(unittest.TestCase):
    def test_forward(self):
        should_be = np.array([1,3,4])
        actually_is = whole_steps(1,6,3)
        for i, (x,y) in enumerate(zip(should_be, actually_is)):
            with self.subTest(i=i):
                self.assertEqual(x,y)
    def test_backward(self):
        should_be = np.array([5,3,2])
        actually_is = whole_steps(5,0,3)
        for i, (x,y) in enumerate(zip(should_be, actually_is)):
            with self.subTest(i=i):
                self.assertEqual(x,y)
    def test_extra(self):
        should_be = np.array([1,2,3,4,5])
        actually_is = whole_steps(1,6,9)
        for i, (x,y) in enumerate(zip(should_be, actually_is)):
            with self.subTest(i=i):
                self.assertEqual(x,y)
    def test_empty(self):
        should_be = np.array([])
        actually_is = whole_steps(1,1,3)
        for i, (x,y) in enumerate(zip(should_be, actually_is)):
            with self.subTest(i=i):
                self.assertEqual(x,y)
        
class EnergyTestCase(unittest.TestCase):
    def setUp(self):
        self.N = 5
        self.noise = audio.white_noise(self.N)
    def test_rms(self):
        self.assertAlmostEqual(rms(self.noise), np.sqrt(np.mean(self.noise**2)))
    def test_bad_length(self):
        with self.assertWarns(RuntimeWarning):
            scale_to_unit_energy(self.noise[0:0])
    def test_division_by_zero(self):
        with self.assertWarns(RuntimeWarning):
            self.assertTrue(np.isnan(sum(scale_to_unit_energy(np.zeros(self.N)))))
    def test_good_energy(self):
        tmp = scale_to_unit_energy(self.noise)
        self.assertAlmostEqual(sum(tmp), np.nansum(tmp))
        self.assertAlmostEqual(rms(tmp), 1.0)

# weighted_divergence_computer.py

from .weighted_divergence_computer import *

class WeightedDivergenceComputerTestCase(unittest.TestCase):
    def setUp(self):
        self.wdc = WeightedDivergenceComputer(16,6,4)
    def test_steps(self):
        for i, (x,y) in enumerate(zip(self.wdc.steps(), [1,4,6,8,11,14])):
            with self.subTest(i=i):
                self.assertEqual(x,y)
    def test_starts(self):
        for i, (x,y) in enumerate(zip(self.wdc.starts(), [1,5,8,12])):
            with self.subTest(i=i):
                self.assertEqual(x,y)
    def test_FFT_size_mismatch(self):
        spg = np.tile(0.0,[50,self.wdc.FFTsize()])
        with self.assertRaises(ValueError):
            self.wdc.update(spg)
    def test_too_few_time_steps(self):
        spg = np.tile(0.0,[self.wdc.FFTsize()//2, self.wdc.FFTsize()//2])
        with self.assertRaises(ValueError):
            self.wdc.update(spg)
    def test_noise(self):
        nr = int(1.5*self.wdc.FFTsize())
        nc = self.wdc.FFTsize() // 2
        spg = np.random.normal(0, 1, [nr,nc])
        self.wdc.update(spg)
        self.assertTrue((self.wdc.values() > 0).all())
        for i, x in enumerate(self.wdc.values()):
            with self.subTest(i=i):
                self.assertNotEqual(x[0],x[1]*x[2])
    def test_fake(self):
        A = np.array([[1, 2, 3, 4, 5, 4, 3, 2],
                      [5, 4, 3, 2, 4, 3, 2, 1],
                      [4, 3, 2, 1, 3, 2, 1, 0],
                      [3, 2, 1, 0, 3, 2, 1, 0],
                      [3, 2, 1, 0, 1, 2, 3, 4],
                      [1, 2, 3, 4, 2, 3, 4, 1],
                      [2, 3, 4, 1, 3, 4, 5, 6],
                      [3, 4, 5, 6, 7, 8, 9, 2],
                      [7, 8, 9, 2, 4, 5, 6, 9],
                      [4, 5, 6, 9, 1, 2, 3, 4],
                      [2, 6, 3, 1, 5, 2, 7, 0],
                      [9, 0, 1, 1, 2, 3, 1, 5],
                      [2, 7, 0, 0, 2, 8, 5, 7],
                      [1, 4, 3, 0, 5, 4, 0, 6],
                      [8, 9, 6, 6, 0, 2, 2, 0],
                      [6, 8, 9, 6, 6, 2, 6, 3]])
        weights = np.array([6.6771005, 1.27792, 0.84497007, 0.3930976, 0.23998331, 0.21444444])
        divs = np.array([83.96624353, 14.01994903, 11.39576704, 4.34260542, 2.56173769, 2.42383993])
        wds = np.array([68.76892309, 11.85760605, 9.71729962, 3.43289668, 2.45909712, 2.07911603])
        self.wdc.update(A)
        for i, x in enumerate(self.wdc.values()):
            with self.subTest(i=i):
                self.assertAlmostEqual(weights[i],x[0])
                self.assertAlmostEqual(divs[i], x[1])
                self.assertAlmostEqual(wds[i], x[2])

if __name__ == '__main__':
    unittest.main()
