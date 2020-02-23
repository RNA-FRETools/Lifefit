import unittest
import lifefit as lf

class HoribaImportTest(unittest.TestCase):

    def testMissingHeader(self):
        fluor, timestep = lf.tcspc.read_decay(lf._TEST_DIR+'/testdata/missing_header.txt')
        self.assertRaises(TypeError)

    def testMissingTimestep(self):
        fluor, timestep = lf.tcspc.read_decay(lf._TEST_DIR+'/testdata/missing_timestep.txt')
        self.assertRaises(TypeError)

    def testTimestep(self):
        fluor, timestep = lf.tcspc.read_decay(lf._DATA_DIR+'/lifetime/Atto550_DNA.txt')
        self.assertAlmostEqual(timestep, 0.02743484)

class LifeFitTest(unittest.TestCase):
    def setUp(self):
        fluor, timestep = lf.tcspc.read_decay(lf._DATA_DIR+'/lifetime/Atto550_DNA.txt')
        irf, _ = tcspc.read_decay(lf._DATA_DIR+'/IRF/irf.txt')



if __name__ == "__main__":
    unittest.main()
