#!/usr/bin/env python
# coding: utf-8

import unittest
from src.stareg.tensorproductspline import TensorProductSpline
import numpy as np


class TestTensorProductSpline(unittest.TestCase):

    def setUp(self):
        self.x = np.random.random(200).reshape(100,2)
        self.x.sort(axis=0)
        self.k1 = 5
        self.k2 = 5
        self.m = 2
        self.TS = TensorProductSpline()

    def tearDown(self):
        del self.TS

    def test_tensorproductspline_basis_quantile(self):
        self.TS.tensor_product_spline_2d_basis(x_data=self.x, k1=self.k1, k2=self.k2, type_="quantile")

        self.assertEqual(self.TS.basis_x1.shape, (self.x.shape[0], self.k1))
        self.assertEqual(self.TS.basis_x2.shape, (self.x.shape[0], self.k2))
        self.assertEqual(self.TS.basis.shape, (self.x.shape[0], self.k1*self.k2))
        self.assertEqual(self.TS.k1, self.k1)
        self.assertEqual(self.TS.k2, self.k2)

    def test_tensorproductspline_basis_equidistant(self):
        self.TS.tensor_product_spline_2d_basis(x_data=self.x, k1=self.k1, k2=self.k2, type_="equidistant")

        self.assertEqual(self.TS.basis_x1.shape, (self.x.shape[0], self.k1))
        self.assertEqual(self.TS.basis_x2.shape, (self.x.shape[0], self.k2))
        self.assertEqual(self.TS.basis.shape, (self.x.shape[0], self.k1*self.k2))
        self.assertEqual(self.TS.k1, self.k1)
        self.assertEqual(self.TS.k2, self.k2)

    def test_tensorproductspline_basis_partition_of_unity(self):
        self.TS.tensor_product_spline_2d_basis(x_data=self.x, k1=self.k1, k2=self.k2, type_="quantile")
        self.assertTrue(np.allclose(self.TS.basis.sum(axis=1), np.ones(len(self.x))))
        self.TS.tensor_product_spline_2d_basis(x_data=self.x, k1=self.k1, k2=self.k2, type_="equidistant")
        self.assertTrue(np.allclose(self.TS.basis.sum(axis=1), np.ones(len(self.x))))

        
    def test_tensorproductspline_basis_16_splines_per_knot_interval(self):
        # first and last spline interval only has 9 nonzeros because first and last spline is 0 at first and last position
        self.TS.tensor_product_spline_2d_basis(x_data=self.x, k1=self.k1, k2=self.k2, type_="equidistant")
        self.assertTrue(((self.TS.basis > 0).sum(axis=1)[1:-1] == 16).all())
        self.TS.tensor_product_spline_2d_basis(x_data=self.x, k1=self.k1, k2=self.k2, type_="quantile")
        self.assertTrue(((self.TS.basis > 0).sum(axis=1)[1:-1] == 16).all())

if __name__ == "__main__":
    
    unittest.main()