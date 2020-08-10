
import unittest
from stareg.penalty_matrix import PenaltyMatrix
from stareg.bspline import Bspline
import numpy as np
from scipy.signal import find_peaks

class TestPenaltyMatrix(unittest.TestCase):

    def setUp(self):
        self.n_param = 25
        self.PM = PenaltyMatrix()

    def tearDown(self):
        del self.PM

    def test_d1_difference_matrix(self): 
        d1 = self.PM.d1_difference_matrix(n_param=self.n_param)
        self.assertEqual(d1.shape, (self.n_param-1, self.n_param))
        self.assertTrue((((d1 == -1).sum(axis=1) + (d1 == 1).sum(axis=1)) == 2).all())
        self.assertTrue(d1[0,0], -1)
        self.assertTrue(d1[0,1], 1)
        

    def test_d2_difference_matrix(self): 
        d2 = self.PM.d2_difference_matrix(n_param=self.n_param)
        self.assertEqual(d2.shape, (self.n_param-2, self.n_param))
        self.assertTrue((((d2 == -2).sum(axis=1) + (d2 == 1).sum(axis=1)) == 3).all())
        self.assertTrue(d2[0,0], 1)
        self.assertTrue(d2[0,1], -2)
        self.assertTrue(d2[0,2], 1)
        
    def test_smoothness_matrix(self): 
        sm = self.PM.smoothness_matrix(n_param=self.n_param)
        self.assertEqual(sm.shape, (self.n_param-2, self.n_param))
        self.assertTrue((((sm == -2).sum(axis=1) + (sm == 1).sum(axis=1)) == 3).all())
        self.assertTrue(sm[0,0], 1)
        self.assertTrue(sm[0,1], -2)
        self.assertTrue(sm[0,2], 1)

    def test_peak(self): 
        x = np.linspace(0, 1, 100)
        y = 0.5*np.exp(-(x - 0.4)**2 / 0.01)
        
        bs = Bspline()
        bs.bspline_basis(x_data=x, k=self.n_param, m=2, type_="equidistant")
        peak = self.PM.peak_matrix(n_param=self.n_param, y_data=y, basis=bs.basis)

        self.assertEqual(peak.shape, (self.n_param-1, self.n_param))
        self.assertEqual(np.count_nonzero(np.count_nonzero(peak, axis=1)==0), 1)
        self.assertTrue(peak[0,0], -1)
        self.assertTrue(peak[0,1], 1)
        self.assertTrue(peak[-1,-1], 1)
        self.assertTrue(peak[-1,-2], 1)

    def test_valley(self): 
        x = np.linspace(0, 1, 100)
        y = -1*0.5*np.exp(-(x - 0.4)**2 / 0.01)
        
        bs = Bspline()
        bs.bspline_basis(x_data=x, k=self.n_param, m=2, type_="equidistant")
        valley = self.PM.valley_matrix(n_param=self.n_param, y_data=y, basis=bs.basis)

        self.assertEqual(valley.shape, (self.n_param-1, self.n_param))
        self.assertEqual(np.count_nonzero(np.count_nonzero(valley, axis=1)==0), 1)
        self.assertTrue(valley[0,0], 1)
        self.assertTrue(valley[0,1], -1)
        self.assertTrue(valley[-1,-1], -1)
        self.assertTrue(valley[-1,-2], 1)


    def test_multi_peak(self):
        x = np.linspace(0, 1, 100)
        y = np.exp(-(x - 0.4)**2 / 0.01) + np.exp(-(x-0.8)**2 / 0.01)
        
        bs = Bspline()
        bs.bspline_basis(x_data=x, k=self.n_param, m=2, type_="equidistant")
        peaks = self.PM.multi_peak_matrix(n_param=self.n_param, y_data=y, basis=bs.basis)
        
        self.assertEqual(peaks.shape, (self.n_param-1, self.n_param))
        self.assertEqual(np.count_nonzero(np.count_nonzero(peaks, axis=1)==0), 3)

    def test_multi_valley(self):
        x = np.linspace(0, 1, 100)
        y = np.exp(-(x - 0.4)**2 / 0.01) + np.exp(-(x-0.8)**2 / 0.01)
        y = -1*y

        bs = Bspline()
        bs.bspline_basis(x_data=x, k=self.n_param, m=2, type_="equidistant")
        valley = self.PM.multi_valley_matrix(n_param=self.n_param, y_data=y, basis=bs.basis)
        
        self.assertEqual(valley.shape, (self.n_param-1, self.n_param))
        self.assertEqual(np.count_nonzero(np.count_nonzero(valley, axis=1)==0), 3)
        
    def test_multi_extremum_peak_then_valley(self):
        x = np.linspace(0, 1, 100)
        y = np.exp(-(x - 0.4)**2 / 0.01) + -1*np.exp(-(x-0.8)**2 / 0.01)
        
        bs = Bspline()
        bs.bspline_basis(x_data=x, k=self.n_param, m=2, type_="equidistant")
        valley = self.PM.multi_extremum_matrix(n_param=self.n_param, y_data=y, basis=bs.basis)
        
        self.assertEqual(valley.shape, (self.n_param-1, self.n_param))
        self.assertEqual(np.count_nonzero(np.count_nonzero(valley, axis=1)==0), 3)

    def test_multi_extremum_valley_then_peak(self):
        x = np.linspace(0, 1, 100)
        y = -1*np.exp(-(x - 0.4)**2 / 0.01) + np.exp(-(x-0.8)**2 / 0.01)
        
        bs = Bspline()
        bs.bspline_basis(x_data=x, k=self.n_param, m=2, type_="equidistant")
        valley = self.PM.multi_extremum_matrix(n_param=self.n_param, y_data=y, basis=bs.basis)
        
        self.assertEqual(valley.shape, (self.n_param-1, self.n_param))
        self.assertEqual(np.count_nonzero(np.count_nonzero(valley, axis=1)==0), 3)


if __name__ == "__main__":
    
    unittest.main()