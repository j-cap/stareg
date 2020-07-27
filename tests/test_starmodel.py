#!/usr/bin/env python
# coding: utf-8


import unittest
from src.stareg.star_model import StarModel
import numpy as np

class TestStarModel(unittest.TestCase):

    def setUp(self):

        x = np.random.random(100)
        x.sort()
        self.x = x.reshape(len(x), 1)
        self.y = 0.5*np.exp(-(self.x - 0.4)**2 / 0.01)
        self.n_params = 25
        self.descr_str =( ("s(1)", "smooth", 25, (1, 100), "equidistant"), )
        self.M = StarModel(descr=self.descr_str)

    def tearDown(self):
        del self.M

    def test_starmodel_create_basis(self):
        self.M.create_basis(X=self.x)
        self.assertEqual(len(self.M.smooths), 1)
        self.assertEqual(self.M.basis.shape, (len(self.x), self.n_params))
        self.assertEqual(self.M.smoothness_penalty_matrix.shape, (self.n_params, self.n_params))

    def test_starmodel_create_constraint_penalty_matrix_1d(self):
        self.M.create_basis(X=self.x)
        self.M.create_constraint_penalty_matrix(beta_test=np.zeros(self.n_params), y=self.y)
        self.assertEqual(self.M.constraint_penalty_matrix.shape, (self.n_params, self.n_params))

    def test_starmodel_create_constraint_penalty_matrix_2d(self):
        self.constraint_pen_mat_2d(c1="smooth", c2="inc")
        self.constraint_pen_mat_2d(c1="smooth", c2="dec")
        self.constraint_pen_mat_2d(c1="peak", c2="smooth", y=self.y.ravel()) 
        self.constraint_pen_mat_2d(c1="valley", c2="smooth", y=-1*self.y.ravel())
        self.constraint_pen_mat_2d(c1="inc", c2="dec")
        self.constraint_pen_mat_2d(c1="peak", c2="inc", y=self.y.ravel()) 
        self.constraint_pen_mat_2d(c1="valley", c2="inc", y=-1*self.y.ravel())
        self.constraint_pen_mat_2d(c1="peak", c2="dec", y=self.y.ravel())
        self.constraint_pen_mat_2d(c1="valley", c2="dec", y=-1*self.y.ravel())
        
    def constraint_pen_mat_2d(self, c1="smooth", c2="smooth", y=None):
        if y is None: y=self.y
        descr_str =( ("s(1)", c1, 25, (1, 100), "equidistant"),
                     ("s(2)", c2, 25, (1, 100), "quantile"),  )
        x = np.random.random(200).reshape(100,2)
        x.sort(axis=0)
        M = StarModel(descr=descr_str)
        M.create_basis(X=x, y=y)
        M.create_constraint_penalty_matrix(beta_test=np.zeros(descr_str[0][2] + descr_str[1][2]), y=y)
        self.assertEqual(M.constraint_penalty_matrix.shape, (2*descr_str[0][2], 2*descr_str[1][2]))
        del M

    def test_starmodel_create_basis_for_prediction_1d(self):
        self.M.create_basis_for_prediction(X=self.x)
        self.assertEqual(self.M.basis_for_prediction.shape, (len(self.x), self.n_params))

    def test_starmodel_create_basis_for_prediction_2d(self):
        descr_str =( ("s(1)", "smooth", 25, (1, 100), "equidistant"),
                     ("s(2)", "smooth", 25, (1, 100), "quantile"),  )
        x = np.random.random(200).reshape(100,2)
        x.sort(axis=0)
        M = StarModel(descr=descr_str)
        M.create_basis_for_prediction(X=x)
        self.assertEqual(M.basis_for_prediction.shape, (x.shape[0], 50))

    def test_starmodel_calc_LS_fit(self):
        self.M = self.M.calc_LS_fit(X=self.x, y=self.y)
        self.assertEqual(len(self.M.coef_), self.M.basis.shape[1])
        self.assertEqual(len(self.M.LS_coef_), self.M.basis.shape[1])
        
    def test_starmodel_calc_LS_fit_2d(self):
        
        descr_str =( ("s(1)", "smooth", 25, (1, 100), "equidistant"),
                     ("s(2)", "smooth", 25, (1, 100), "quantile"),  )
        x = np.random.random(200).reshape(100,2)
        x.sort(axis=0)
        y = np.sin(x[:,0]) + x[:,0]*x[:,1]
        M = StarModel(descr=descr_str)
        M.calc_LS_fit(X=x, y=y)
        
        self.assertEqual(len(M.coef_), M.basis.shape[1])
        self.assertEqual(len(M.LS_coef_), M.basis.shape[1])
        self.assertEqual(len(M.coef_), 50)
        self.assertEqual(len(M.LS_coef_), 50)

    def test_starmodel_create_df_for_beta(self):
        beta_test = np.random.random(self.n_params)
        df = self.M.create_df_for_beta(beta_init=beta_test)
        self.assertEqual(df.shape, (1, self.n_params))
        self.assertTrue(np.allclose(df.values, beta_test))
        
    def test_starmodel_fit_1d(self):
        self.M = self.M.fit(X=self.x, y=self.y, plot_=False)
        self.assertEqual(len(self.M.coef_), self.M.basis.shape[1])

    def test_starmodel_fit_2d(self):
        descr_str =( ("s(1)", "smooth", 25, (1, 100), "equidistant"),
                     ("s(2)", "smooth", 25, (1, 100), "quantile"),  )
        x = np.random.random(200).reshape(100,2)
        x.sort(axis=0)
        y = np.sin(x[:,0]) + x[:,0]*x[:,1]
        M = StarModel(descr=descr_str)
        M = M.fit(X=x, y=y, plot_=False)
        self.assertEqual(len(M.coef_), M.basis.shape[1])

    def test_starmodel_predict(self):
        self.M.fit(X=self.x, y=self.y, plot_=False)
        pred = self.M.predict(X=np.array([0.2]))
        self.assertEqual(len(pred), 1)
        self.assertEqual(pred, float, msg="Fix the predict method!!!")


if __name__ == "__main__":
    
    unittest.main()