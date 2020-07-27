#!/usr/bin/env python
# coding: utf-8

import unittest
from src.stareg.smooth import Smooths, TensorProductSmooths
import numpy as np


class TestSmooths(unittest.TestCase):


    def setUp(self):
        x = np.random.random(100)
        x.sort
        self.x = x
        self.y = 0.5*np.exp(-(self.x - 0.4)**2 / 0.01)
        self.n_param = 25
        self.S = Smooths(
            x_data=self.x, n_param=self.n_param, 
            constraint="smooth", y_peak_or_valley=None, 
            lambdas=None, type_="quantile")

    def tearDown(self):
        del self.S

    def test_Smooths(self):
        self.assertEqual(self.S.basis.shape, (len(self.x), self.n_param))
        self.assertEqual(Smooths(x_data=self.x, n_param=self.n_param, constraint="inc").penalty_matrix.shape, (self.n_param -1, self.n_param))
        self.assertEqual(Smooths(x_data=self.x, n_param=self.n_param, constraint="dec").penalty_matrix.shape, (self.n_param -1, self.n_param))
        self.assertEqual(Smooths(x_data=self.x, n_param=self.n_param, constraint="conv").penalty_matrix.shape, (self.n_param -2, self.n_param))
        self.assertEqual(Smooths(x_data=self.x, n_param=self.n_param, constraint="conv").penalty_matrix.shape, (self.n_param -2, self.n_param))
        self.assertEqual(Smooths(x_data=self.x, n_param=self.n_param, constraint="peak", y_peak_or_valley=self.y).penalty_matrix.shape, (self.n_param - 1, self.n_param))
        self.assertEqual(Smooths(x_data=self.x, n_param=self.n_param, constraint="valley", y_peak_or_valley=-1*self.y).penalty_matrix.shape, (self.n_param - 1, self.n_param))

    def test_Smooths_constraint_penalty_matrices(self):
        inc = Smooths(x_data=self.x, n_param=self.n_param, constraint="inc", y_peak_or_valley=None).penalty_matrix
        self.assertTrue((((inc == -1).sum(axis=1) + (inc == 1).sum(axis=1)) == 2).all())
        self.assertTrue(inc[0,0], -1)
        self.assertTrue(inc[0,1], 1)
        dec = Smooths(x_data=self.x, n_param=self.n_param, constraint="dec", y_peak_or_valley=None).penalty_matrix 
        self.assertTrue((((dec == -1).sum(axis=1) + (dec == 1).sum(axis=1)) == 2).all())
        self.assertTrue(dec[0,0], -1)
        self.assertTrue(dec[0,1], 1)
        conv = Smooths(x_data=self.x, n_param=self.n_param, constraint="conv", y_peak_or_valley=None).penalty_matrix
        self.assertTrue((((conv == 1).sum(axis=1) + (conv == -2).sum(axis=1)) == 3).all())
        self.assertTrue(conv[0,0], 1)
        self.assertTrue(conv[0,1], -2)
        self.assertTrue(conv[0,2], 1)
        conc = Smooths(x_data=self.x, n_param=self.n_param, constraint="conc", y_peak_or_valley=None).penalty_matrix
        self.assertTrue((((conc == -1).sum(axis=1) + (conc == 2).sum(axis=1)) == 3).all())
        self.assertTrue(conc[0,0], -1)
        self.assertTrue(conc[0,1], 2)
        self.assertTrue(conc[0,2], -1)
 
class TestTensorProductSmooths(unittest.TestCase):

    def setUp(self):
        
        x = np.random.random(200).reshape((100,2))
        x.sort(axis=0)
        self.x = x 
        self.k1, self.k2 = 5, 5
        self.TS = TensorProductSmooths(x_data=self.x, n_param=(self.k1, self.k2), constraint="smooth", lambdas=None, type_="quantile")

    def tearDown(self):
        del self.TS

    def test_TensorProductSmooths(self):
        self.assertEqual(self.TS.basis.shape, (len(self.x), self.k1 * self.k2))
        self.assertEqual(self.TS.penalty_matrix.shape, ((self.k1 * self.k2)-2, self.k1 * self.k2))


if __name__ == "__main__":
    
    unittest.main()

