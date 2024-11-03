#!/usr/bin/env python3

import unittest
import lifefit as lf
import requests
import io
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent


class HoribaImportTest(unittest.TestCase):

    def testMissingHeader(self):
        with self.assertRaises(ValueError):
            fluor, timestep = lf.tcspc.read_decay(TEST_DIR.joinpath("testdata", "missing_header.txt"))

    def testMissingTimestep(self):
        with self.assertRaises(ValueError):
            fluor, timestep = lf.tcspc.read_decay(TEST_DIR.joinpath("testdata", "missing_timestep.txt"))

    def testTimestep(self):
        fluor, timestep = lf.tcspc.read_decay(lf.DATA_DIR.joinpath("lifetime", "Atto550_DNA.txt"))
        self.assertAlmostEqual(timestep, 0.0274, places=4)

    def testTimestep(self):
        fluor, timestep = lf.tcspc.read_decay(
            lf.DATA_DIR.joinpath("lifetime", "Atto550_DNA_time_intensity.txt"), fileformat="time_intensity"
        )
        self.assertAlmostEqual(timestep, 0.0274, places=4)

    def testLoadfromURL(self):
        datastr = requests.get(
            "https://raw.githubusercontent.com/RNA-FRETools/Lifefit/master/src/lifefit/data/lifetime/Atto550_DNA.txt"
        )
        file = io.StringIO(datastr.text)
        fluor, timestep = lf.tcspc.read_decay(file)
        self.assertAlmostEqual(timestep, 0.0274, places=4)


class LifeFitTest(unittest.TestCase):
    def setUp(self):
        pass

    def testReconvolutionFit_withExpIRF(self):
        self.atto550_dna_life = lf.tcspc.Lifetime.from_filenames(
            lf.DATA_DIR.joinpath("lifetime", "Atto550_DNA.txt"), lf.DATA_DIR.joinpath("IRF", "irf.txt")
        )
        self.atto550_dna_life.reconvolution_fit([1, 5], verbose=False)
        self.assertEqual(self.atto550_dna_life.irf_type, "experimental")
        self.assertEqual(len(self.atto550_dna_life.fit_param["tau"]), 2)
        self.assertAlmostEqual(self.atto550_dna_life.av_lifetime, 3.6, places=1)

    def testReconvolutionFit_withGaussIRF(self):
        self.atto550_dna_life = lf.tcspc.Lifetime.from_filenames(lf.DATA_DIR.joinpath("lifetime", "Atto550_DNA.txt"))
        self.atto550_dna_life.reconvolution_fit([3], verbose=False)
        self.assertEqual(self.atto550_dna_life.irf_type, "Gaussian")
        self.assertEqual(len(self.atto550_dna_life.fit_param["tau"]), 1)
        self.assertEqual(self.atto550_dna_life.gauss_sigma, 0.01)
        self.assertEqual(self.atto550_dna_life.gauss_amp, 10000)

    def testReconvolutionFit_withGaussIRF_ampFixed(self):
        self.atto550_dna_life = lf.tcspc.Lifetime.from_filenames(
            lf.DATA_DIR.joinpath("lifetime", "Atto550_DNA.txt"), gauss_amp=10001
        )
        self.atto550_dna_life.reconvolution_fit([3], verbose=False)
        self.assertEqual(self.atto550_dna_life.gauss_amp, 10001)

    def testReconvolutionFit_withGaussIRF_sigmaFixed(self):
        self.atto550_dna_life = lf.tcspc.Lifetime.from_filenames(
            lf.DATA_DIR.joinpath("lifetime", "Atto550_DNA.txt"), gauss_sigma=0.012
        )
        self.atto550_dna_life.reconvolution_fit([3], verbose=False)
        self.assertEqual(self.atto550_dna_life.gauss_sigma, 0.012)
        self.assertAlmostEqual(self.atto550_dna_life.av_lifetime, 3.6, places=1)


class AnisotropyTest(unittest.TestCase):
    def setUp(self):
        pass

    def testOneRotationFit_withExpIRF(self):
        self.atto550_dna_life = {}
        for c in ["VV", "VH", "HV", "HH"]:
            self.atto550_dna_life[c] = lf.tcspc.Lifetime.from_filenames(
                lf.DATA_DIR.joinpath("anisotropy", f"{c}.txt"), lf.DATA_DIR.joinpath("IRF", "irf.txt")
            )
        self.atto550_dna_aniso = lf.tcspc.Anisotropy(
            self.atto550_dna_life["VV"],
            self.atto550_dna_life["VH"],
            self.atto550_dna_life["HV"],
            self.atto550_dna_life["HH"],
        )
        self.atto550_dna_aniso.rotation_fit(p0=[0.4, 5], model="one_rotation")
        self.assertAlmostEqual(self.atto550_dna_aniso.fit_param[1], 5, places=0)


if __name__ == "__main__":
    unittest.main()
