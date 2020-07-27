
import unittest
from src.stareg.penalty_matrix import  PenaltyMatrix
from src.stareg.bspline import Bspline
import numpy as np
from scipy.signal import find_peaks

class TestPenaltyMatrix(unittest.TestCase):

    def setUp(self):
        self.n_param = 25
        self.PM = PenaltyMatrix(n_param=self.n_param)

    def tearDown(self):
        del self.PM

    def test_d1_difference_matrix(self): 
        self.d1 = self.PM.d1_difference_matrix()
        self.assertEqual(self.d1.shape, (self.PM.n_param-1, self.PM.n_param))
        self.assertTrue((((self.d1 == -1).sum(axis=1) + (self.d1 == 1).sum(axis=1)) == 2).all())
        self.assertTrue(self.d1[0,0], -1)
        self.assertTrue(self.d1[0,1], 1)
        

    def test_d2_difference_matrix(self): 
        self.d2 = self.PM.d2_difference_matrix()
        self.assertEqual(self.d2.shape, (self.PM.n_param-2, self.PM.n_param))
        self.assertTrue((((self.d2 == -2).sum(axis=1) + (self.d2 == 1).sum(axis=1)) == 3).all())
        self.assertTrue(self.d2[0,0], 1)
        self.assertTrue(self.d2[0,1], -2)
        self.assertTrue(self.d2[0,2], 1)
        
    def test_smoothness_matrix(self): 
        self.sm = self.PM.smoothness_matrix()
        self.assertEqual(self.sm.shape, (self.PM.n_param-2, self.PM.n_param))
        self.assertTrue((((self.sm == -2).sum(axis=1) + (self.sm == 1).sum(axis=1)) == 3).all())
        self.assertTrue(self.sm[0,0], 1)
        self.assertTrue(self.sm[0,1], -2)
        self.assertTrue(self.sm[0,2], 1)

    def test_peak(self): 
        x = np.linspace(0, 1, 100)
        y = 0.5*np.exp(-(x - 0.4)**2 / 0.01)
        
        bs = Bspline()
        bs.bspline_basis(x_data=x, k=self.n_param, m=2, type_="equidistant")
        self.peak = self.PM.peak_matrix(y_data=y, basis=bs.basis)

        self.assertEqual(self.peak.shape, (self.PM.n_param-1, self.PM.n_param))
        self.assertEqual(np.count_nonzero(np.count_nonzero(self.peak, axis=1)==0), 4)
        self.assertEqual(np.count_nonzero(np.count_nonzero(self.peak, axis=1)), self.peak.shape[0] - 4)
        self.assertTrue(self.peak[0,0], -1)
        self.assertTrue(self.peak[0,1], 1)
        self.assertTrue(self.peak[-1,-1], 1)
        self.assertTrue(self.peak[-1,-2], 1)

    def test_valley(self): 
        x = np.linspace(0, 1, 100)
        y = -1*0.5*np.exp(-(x - 0.4)**2 / 0.01)
        
        bs = Bspline()
        bs.bspline_basis(x_data=x, k=self.n_param, m=2, type_="equidistant")
        self.valley = self.PM.valley_matrix(y_data=y, basis=bs.basis)

        self.assertEqual(self.valley.shape, (self.PM.n_param-1, self.PM.n_param))
        self.assertEqual(np.count_nonzero(np.count_nonzero(self.valley, axis=1)==0), 4)
        self.assertEqual(np.count_nonzero(np.count_nonzero(self.valley, axis=1)), self.valley.shape[0] - 4)
        self.assertTrue(self.valley[0,0], 1)
        self.assertTrue(self.valley[0,1], -1)
        self.assertTrue(self.valley[-1,-1], -1)
        self.assertTrue(self.valley[-1,-2], 1)

if __name__ == "__main__":
    
    unittest.main()