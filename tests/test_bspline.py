#!/usr/bin/env python
# coding: utf-8

import unittest
import numpy as np
from stareg.bspline import Bspline


class TestBspline(unittest.TestCase):

    def setUp(self):
        self.x = np.random.random(100)
        self.x.sort()
        self.k = 10
        self.m = 2
        self.BS = Bspline()
        
    def tearDown(self):
        del self.x 
        del self.BS

    def test_bspline_basis_quantile(self):
        self.BS.bspline_basis(x_data=self.x, k=self.k, m=self.m, type_="quantile")
        self.basis = self.BS.basis
        self.assertEqual(self.basis.shape, (100, 10))
        self.assertTrue(np.allclose(self.BS.knots[3:-3], np.quantile(a=self.x, q=np.linspace(0, 1, self.k-self.m))))
        self.assertTrue(self.BS.n_param == self.k)

    def test_bspline_basis_euqidistant(self):
        self.BS.bspline_basis(x_data=self.x, k=self.k, m=self.m, type_="equidistant")
        self.assertEqual(self.BS.basis.shape, (100, 10))
        self.assertTrue(np.allclose(self.BS.knots[3:-3], np.linspace(self.x.min(), self.x.max(), num=self.k-self.m)))
        self.assertTrue(self.BS.n_param == self.k)
        
    def test_bspline_basis_partition_of_unity(self):
        self.BS.bspline_basis(x_data=self.x, k=self.k, m=self.m, type_="equidistant")
        self.assertTrue(np.allclose(self.BS.basis.sum(axis=1), np.ones(len(self.x))))
        self.BS.bspline_basis(x_data=self.x, k=self.k, m=self.m, type_="quantile")
        self.assertTrue(np.allclose(self.BS.basis.sum(axis=1), np.ones(len(self.x))))
        
    def test_bspline_basis_4_splines_per_knot_interval(self):
        # first and last spline interval only has 3 nonzeros because first and last spline is 0 at first and last position
        self.BS.bspline_basis(x_data=self.x, k=self.k, m=self.m, type_="equidistant")
        self.assertTrue(((self.BS.basis > 0).sum(axis=1)[1:-1] == 4).all())
        self.BS.bspline_basis(x_data=self.x, k=self.k, m=self.m, type_="quantile")
        self.assertTrue(((self.BS.basis > 0).sum(axis=1)[1:-1] == 4).all())
        

if __name__ == "__main__":

    unittest.main()