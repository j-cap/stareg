#!/usr/bin/env python
# coding: utf-8


import unittest
from stareg.star_model import StarModel
import numpy as np

class TestStarModel(unittest.TestCase):

    def setUp(self):


        self.x = np.linspace(0,1,100).reshape(100, 1)
        self.y = np.exp(-(self.x - 0.4)**2 / 0.01)
        self.n_params = 25
        self.descr_str =( ("s(1)", "smooth", 25, (1, 100), "equidistant"), )
        self.M = StarModel(description=self.descr_str)

    def tearDown(self):
        del self.M

    def test_starmodel_create_basis(self):
        self.M.create_basis(X=self.x)
        self.assertEqual(len(self.M.smooths), 1)
        self.assertEqual(self.M.basis.shape, (len(self.x), self.n_params))
        self.assertEqual(self.M.smoothness_penalty_matrix.shape, (self.n_params, self.n_params))

    def test_starmodel_create_constraint_penalty_matrix(self):
        self.M.create_basis(X=self.x, y=self.y)
        self.M.create_constraint_penalty_matrix(beta_test=np.zeros(self.n_params))
        self.assertEqual(self.M.constraint_penalty_matrix.shape, (self.n_params, self.n_params))

    def test_starmodel_create_constraint_penalty_matrix_2d(self):
        self.constraint_pen_mat_2d(c1="smooth", c2="inc")
        self.constraint_pen_mat_2d(c1="smooth", c2="dec")
        self.constraint_pen_mat_2d(c1="peak", c2="smooth") 
        self.constraint_pen_mat_2d(c1="valley", c2="smooth", y=-1*self.y)
        self.constraint_pen_mat_2d(c1="inc", c2="dec")
        self.constraint_pen_mat_2d(c1="peak", c2="inc") 
        self.constraint_pen_mat_2d(c1="valley", c2="inc", y=-1*self.y)
        self.constraint_pen_mat_2d(c1="peak", c2="dec")
        self.constraint_pen_mat_2d(c1="valley", c2="dec", y=-1*self.y)
        
    def constraint_pen_mat_2d(self, c1="smooth", c2="smooth", y=None):
        if y is None: y=self.y
        descr_str =( ("s(1)", c1, 25, (1, 100), "equidistant"),
                     ("s(2)", c2, 25, (1, 100), "quantile"),  )
        x = np.random.random(200).reshape(100,2)
        x.sort(axis=0)
        M = StarModel(description=descr_str)
        M.calc_LS_fit(X=x, y=y)
        M.create_constraint_penalty_matrix(beta_test=M.coef_)
        self.assertEqual(M.constraint_penalty_matrix.shape, (2*descr_str[0][2], 2*descr_str[1][2]))
        del M

    def test_starmodel_create_basis_for_prediction(self):
        self.M.create_basis_for_prediction(X=self.x)
        self.assertEqual(self.M.basis_for_prediction.shape, (len(self.x), self.n_params))

    def test_starmodel_create_basis_for_prediction_2d(self):
        descr_str =( ("s(1)", "smooth", 25, (1, 100), "equidistant"),
                     ("s(2)", "smooth", 25, (1, 100), "quantile"),  )
        x = np.random.random(200).reshape(100,2)
        x.sort(axis=0)
        M = StarModel(description=descr_str)
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
        M = StarModel(description=descr_str)
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
        
    def test_starmodel_fit(self):
        self.M = self.M.fit(X=self.x, y=self.y, plot_=False)
        self.assertEqual(len(self.M.coef_), self.M.basis.shape[1])

    def test_starmodel_fit_2d(self):
        descr_str =( ("s(1)", "smooth", 25, (1, 100), "equidistant"),
                     ("s(2)", "smooth", 25, (1, 100), "quantile"), 
                     ("t(1,2)", "smooth", (5,5), (1, 100), "quantile"), )
        x = np.random.random(200).reshape(100,2)
        x.sort(axis=0)
        y = np.sin(x[:,0]) + x[:,0]*x[:,1]
        M = StarModel(description=descr_str)
        M = M.fit(X=x, y=y, plot_=False)
        self.assertEqual(len(M.coef_), M.basis.shape[1])

    def test_starmodel_predict(self):
        # predict is not finished
        pass


    def test_starmodel_calc_hat_matrix(self):
        self.M.fit(X=self.x, y=self.y, plot_=False)
        H = self.M.calc_hat_matrix()
        self.assertEqual(H.shape, (self.x.shape[0], self.x.shape[0]))
        self.assertTrue(np.allclose(H, H.T))
    
    def test_starmodel_calc_hat_matrix_2d(self):
        descr_str =( ("s(1)", "smooth", 25, (1, 100), "equidistant"),
                     ("s(2)", "smooth", 25, (1, 100), "quantile"),  )
        x = np.random.random(200).reshape(100,2)
        x.sort(axis=0)
        y = np.sin(x[:,0]) + x[:,0]*x[:,1]
        M = StarModel(description=descr_str)
        M = M.fit(X=x, y=y, plot_=False)
        H = M.calc_hat_matrix()
        self.assertEqual(H.shape, (x.shape[0], x.shape[0]))
        self.assertTrue(np.allclose(H, H.T))

    def test_starmodel_calc_GCV_score(self):
        self.M.fit(X=self.x, y=self.y, plot_=False)
        gcv = self.M.calc_GCV_score(y=self.y)
        self.assertEqual(type(gcv), np.float64)

    def test_starmodel_calc_GCV_score_2d(self):
        descr_str =( ("s(1)", "smooth", 25, (1, 100), "equidistant"),
                     ("s(2)", "smooth", 25, (1, 100), "quantile"),  )
        x = np.random.random(200).reshape(100,2)
        x.sort(axis=0)
        y = np.sin(x[:,0]) + x[:,0]*x[:,1]
        M = StarModel(description=descr_str)
        M = M.fit(X=x, y=y, plot_=False)
        gcv = M.calc_GCV_score(y=y)
        self.assertEqual(type(gcv), np.float64)

    def test_starmodel_generate_GCV_parameter_list(self):
        n_grid = 5
        p_min = 1e-4
        grid = self.M.generate_GCV_parameter_list(n_grid=n_grid, p_min=p_min)
        i = 0
        c = np.inf
        for _, g in enumerate(grid): 
            i += 1
            c1, c2 = g["s(1)_constraint"], g["s(1)_smoothness"]
            cc = np.min([c1, c2])
            c = np.min([c, cc])

        self.assertEqual(i, n_grid**2)
        self.assertEqual(c, p_min)

    def test_starmodel_generate_GCV_parameter_list_2d(self):
        descr_str =( ("s(1)", "smooth", 25, (1, 100), "equidistant"),
                     ("s(2)", "smooth", 25, (1, 100), "quantile"),  )
        x = np.random.random(200).reshape(100,2)
        x.sort(axis=0)
        M = StarModel(description=descr_str)
        n_grid, p_min = 5, 1e-4
        grid = M.generate_GCV_parameter_list(n_grid=n_grid, p_min=p_min)
        i, c = 0, np.inf
        for _, g in enumerate(grid): 
            i += 1
            c1, c2 = g["s(1)_constraint"], g["s(1)_smoothness"]
            cc = np.min([c1, c2])
            c = np.min([c, cc])

        self.assertEqual(i, n_grid**4)
        self.assertEqual(c, p_min)

    def test_starmodel_get_params(self):
        d = self.M.get_params()
        self.assertTrue(type(d) == dict)
        self.assertTrue(type(d["s(1)"]) == dict)
        self.assertTrue(len(d["s(1)"].keys()) == 4)

    def test_starmodel_get_params_2d(self):
        descr_str =( ("s(1)", "smooth", 25, (1, 100), "equidistant"),
                     ("s(2)", "smooth", 25, (1, 100), "quantile"),  )
        x = np.random.random(200).reshape(100,2)
        x.sort(axis=0)
        M = StarModel(description=descr_str)
        d = M.get_params()
        self.assertTrue(type(d) == dict)
        self.assertTrue(len(d.keys()) == 2)
        self.assertTrue(type(d["s(1)"]) == dict)
        self.assertTrue(len(d["s(1)"].keys()) == 4)

    def test_starmodel_set_params(self):
        self.M.create_basis(X=self.x)
        d = self.M.get_params()
        d["s(1)"]["constraint"] = "CHANGED"     
        d["s(1)"]["knot_type"] = "CHANGED"     
        d["s(1)"]["lam"] = {"smoothness": "CHANGED", "constraint": "CHANGED"}     
        d["s(1)"]["n_param"] = "CHANGED"     
        self.M.set_params(params=d)  
        self.assertEqual(self.M.description_dict["s(1)"]["constraint"], "CHANGED")
        self.assertEqual(self.M.description_dict["s(1)"]["knot_type"], "CHANGED")
        self.assertEqual(self.M.description_dict["s(1)"]["lam"], {"smoothness": "CHANGED", "constraint": "CHANGED"})
        self.assertEqual(self.M.description_dict["s(1)"]["n_param"], "CHANGED")

    def test_starmodel_set_params_2d(self):
        descr_str =( ("s(1)", "smooth", 25, (1, 100), "equidistant"),
                     ("s(2)", "smooth", 25, (1, 100), "quantile"),  
                     ("t(1,2)", "smooth", (5,5), (1, 100), "equidistant"), )
        x = np.random.random(200).reshape(100,2)
        x.sort(axis=0)
        M = StarModel(description=descr_str)
        M.create_basis(X=x)
        d = M.get_params()
        d["s(1)"]["constraint"] = "CHANGED"     
        d["s(1)"]["knot_type"] = "CHANGED"     
        d["s(1)"]["lam"] = {"smoothness": "CHANGED", "constraint": "CHANGED"}     
        d["s(1)"]["n_param"] = "CHANGED"     
        d["s(2)"]["constraint"] = "CHANGED"     
        d["s(2)"]["knot_type"] = "CHANGED"     
        d["s(2)"]["lam"] = {"smoothness": "CHANGED", "constraint": "CHANGED"}     
        d["s(2)"]["n_param"] = "CHANGED"     
        d["t(1,2)"]["constraint"] = "CHANGED"
        d["t(1,2)"]["knot_type"] = "CHANGED"
        d["t(1,2)"]["lam"] = {"smoothness": "CHANGED", "constraint": "CHANGED"}
        d["t(1,2)"]["n_param"] = "CHANGED"
        M.set_params(params=d)  
        self.assertEqual(M.description_dict["s(1)"]["constraint"], "CHANGED")
        self.assertEqual(M.description_dict["s(1)"]["knot_type"], "CHANGED")
        self.assertEqual(M.description_dict["s(1)"]["lam"], {"smoothness": "CHANGED", "constraint": "CHANGED"})
        self.assertEqual(M.description_dict["s(1)"]["n_param"], "CHANGED")
        self.assertEqual(M.description_dict["s(2)"]["constraint"], "CHANGED")
        self.assertEqual(M.description_dict["s(2)"]["knot_type"], "CHANGED")
        self.assertEqual(M.description_dict["s(2)"]["lam"], {"smoothness": "CHANGED", "constraint": "CHANGED"})
        self.assertEqual(M.description_dict["s(2)"]["n_param"], "CHANGED")
        self.assertEqual(M.description_dict["t(1,2)"]["constraint"], "CHANGED")
        self.assertEqual(M.description_dict["t(1,2)"]["knot_type"], "CHANGED")
        self.assertEqual(M.description_dict["t(1,2)"]["lam"], {"smoothness": "CHANGED", "constraint": "CHANGED"})
        self.assertEqual(M.description_dict["t(1,2)"]["n_param"], "CHANGED")

    def test_starmodel_set_params_after_gcv(self):
        d = dict()
        d["s(1)_constraint"] = "CHANGED"
        d["s(1)_smoothness"] = "CHANGED"     
        self.M.set_params_after_gcv(params=d)
        self.assertEqual(self.M.description_dict["s(1)"]["lam"]["smoothness"], "CHANGED")
        self.assertEqual(self.M.description_dict["s(1)"]["lam"]["constraint"], "CHANGED")

    def test_starmodel_set_params_after_gcv_2d(self):
        descr_str =( ("s(1)", "smooth", 25, (1, 100), "equidistant"),
                     ("s(2)", "smooth", 25, (1, 100), "quantile"),
                     ("t(1,2)", "smooth", (5,5), (1,100), "quantile"), )
        x = np.random.random(200).reshape(100,2)
        x.sort(axis=0)
        M = StarModel(description=descr_str)
        d = dict()
        d["s(1)_constraint"] = "CHANGED"
        d["s(1)_smoothness"] = "CHANGED"     
        d["s(2)_constraint"] = "CHANGED"
        d["s(2)_smoothness"] = "CHANGED"     
        d["t(1,2)_constraint"] = "CHANGED"
        d["t(1,2)_smoothness"] = "CHANGED"
        M.set_params_after_gcv(params=d)
        self.assertEqual(M.description_dict["s(1)"]["lam"]["smoothness"], "CHANGED")
        self.assertEqual(M.description_dict["s(1)"]["lam"]["constraint"], "CHANGED")
        self.assertEqual(M.description_dict["s(2)"]["lam"]["smoothness"], "CHANGED")
        self.assertEqual(M.description_dict["s(2)"]["lam"]["constraint"], "CHANGED")
        self.assertEqual(M.description_dict["t(1,2)"]["lam"]["smoothness"], "CHANGED")
        self.assertEqual(M.description_dict["t(1,2)"]["lam"]["constraint"], "CHANGED")


if __name__ == "__main__":
    
    unittest.main()