

import unittest
from src.stareg.utils import check_constraint, check_constraint_full_model
from src.stareg.utils import check_valley_constraint, check_peak_constraint
from src.stareg.star_model import StarModel
import numpy as np

class TestUtils(unittest.TestCase):

    def setUp(self):
        x = np.random.random(100)
        x.sort()
        self.x = x
        self.y = 2*np.exp(-(x - 0.4)**2 / 0.01)
        self.n_param = 20

    def tearDown(self):
        del self.x
        del self.y

    def test_check_constraint_inc(self):
        # decreasing sequence
        beta = np.linspace(0,-1, self.n_param)
        V = check_constraint(beta=beta, constraint="inc", )
        self.assertEqual(V.shape, (self.n_param-1, self.n_param-1))
        self.assertTrue((np.diag(V) == np.ones(self.n_param-1)).all())
        # increasing sequence
        beta = np.linspace(0,1, self.n_param)
        V = check_constraint(beta=beta, constraint="inc", )
        self.assertEqual(V.shape, (self.n_param-1, self.n_param-1))
        self.assertTrue((np.diag(V) == np.zeros(self.n_param-1)).all())
        # decreasing sequence with increasing part
        beta = np.linspace(0,-1, self.n_param)
        beta[2] = 1
        V = check_constraint(beta=beta, constraint="inc", )
        self.assertEqual(V.shape, (self.n_param-1, self.n_param-1))
        self.assertEqual(V.sum(), self.n_param-2)
        # increasing sequence with decreasing part
        beta = np.linspace(0,1, self.n_param)
        beta[2] = -1
        V = check_constraint(beta=beta, constraint="inc", )
        self.assertEqual(V.shape, (self.n_param-1, self.n_param-1))
        self.assertEqual(V.sum(), 1)

    def test_check_constraint_dec(self):
        # increasing sequence
        beta = np.linspace(0,1, self.n_param)
        V = check_constraint(beta=beta, constraint="dec", )
        self.assertEqual(V.shape, (self.n_param-1, self.n_param-1))
        self.assertTrue((np.diag(V) == np.ones(self.n_param-1)).all())
        # decreasing sequence
        beta = np.linspace(0,-1, self.n_param)
        V = check_constraint(beta=beta, constraint="dec", )
        self.assertEqual(V.shape, (self.n_param-1, self.n_param-1))
        self.assertTrue((np.diag(V) == np.zeros(self.n_param-1)).all())
        # increasing sequence with decreasing part
        beta = np.linspace(0,1,self.n_param)
        beta[2] = -1
        V = check_constraint(beta=beta, constraint="dec", )
        self.assertEqual(V.shape, (self.n_param-1, self.n_param-1))
        self.assertEqual(V.sum(), self.n_param-2)
        # decreasing sequence with increasing part
        beta = np.linspace(0,-1,self.n_param)
        beta[2] = 1
        V = check_constraint(beta=beta, constraint="dec", )
        self.assertEqual(V.shape, (self.n_param-1, self.n_param-1))
        self.assertEqual(V.sum(), 1)
        
    def test_check_constraint_conv(self):
        # convex sequence
        beta = np.exp(np.linspace(0, 2, self.n_param))
        V = check_constraint(beta=beta, constraint="conv")
        self.assertEqual(V.shape, (self.n_param-2, self.n_param-2))
        self.assertTrue((np.diag(V) == np.zeros(self.n_param-2)).all())
        # concave
        beta = -1*np.exp(np.linspace(0, 2, self.n_param))
        V = check_constraint(beta=beta, constraint="conv")
        self.assertEqual(V.shape, (self.n_param-2, self.n_param-2))
        self.assertTrue((np.diag(V) == np.ones(self.n_param-2)).all())
        # convex sequence with concave part
        beta = np.exp(np.linspace(0, 2, self.n_param))
        beta[2] = -1*beta[2]
        V = check_constraint(beta=beta, constraint="conv")
        self.assertEqual(V.shape, (self.n_param-2, self.n_param-2))
        self.assertEqual(V.sum(), 2)
        # concave sequence with convex part
        beta = -1*np.exp(np.linspace(0, 2, self.n_param))
        beta[2] = -1*beta[2]
        V = check_constraint(beta=beta, constraint="conv")
        self.assertEqual(V.shape, (self.n_param-2, self.n_param-2))
        self.assertEqual(V.sum(), self.n_param-2-2)

    def test_check_constraint_conc(self):
        # convex sequence
        beta = np.exp(np.linspace(0, 2, self.n_param))
        V = check_constraint(beta=beta, constraint="conc")
        self.assertEqual(V.shape, (self.n_param-2, self.n_param-2))
        self.assertTrue((np.diag(V) == np.ones(self.n_param-2)).all())
        # concave
        beta = -1*np.exp(np.linspace(0, 2, self.n_param))
        V = check_constraint(beta=beta, constraint="conc")
        self.assertEqual(V.shape, (self.n_param-2, self.n_param-2))
        self.assertTrue((np.diag(V) == np.zeros(self.n_param-2)).all())
        # convex sequence with concave part
        beta = np.exp(np.linspace(0, 2, self.n_param))
        beta[2] = -1*beta[2]
        V = check_constraint(beta=beta, constraint="conc")
        self.assertEqual(V.shape, (self.n_param-2, self.n_param-2))
        self.assertEqual(V.sum(), self.n_param-2-2)
        # concave sequence with convex part
        beta = -1*np.exp(np.linspace(0, 2, self.n_param))
        beta[2] = -1*beta[2]
        V = check_constraint(beta=beta, constraint="conc")
        self.assertEqual(V.shape, (self.n_param-2, self.n_param-2))
        self.assertEqual(V.sum(), 2)
        
    def test_check_constraint_smooth(self):
        # random sequence
        beta = np.random.random(self.n_param)
        V = check_constraint(beta=beta, constraint="smooth")
        self.assertEqual(V.shape, (self.n_param-2, self.n_param-2))
        self.assertTrue((np.diag(V) == np.ones(self.n_param-2)).all())

    def test_check_constraint_peak(self):
        # peak sequence
        beta = np.exp(-(np.linspace(0,1, self.n_param) - 0.4)**2 / 0.01)
        V = check_constraint(beta=beta, constraint="peak")
        self.assertEqual(V.shape, (self.n_param-1, self.n_param-1))
        self.assertTrue(np.array_equal(V, V.astype(bool)))

    def test_check_constraint_valley(self):
        # valley sequence
        beta = -1*np.exp(-(np.linspace(0,1, self.n_param) - 0.4)**2 / 0.01)
        V = check_constraint(beta=beta, constraint="valley")
        self.assertEqual(V.shape, (self.n_param-1, self.n_param-1))
        self.assertTrue(np.array_equal(V, V.astype(bool)))



if __name__ == "__main__":
    unittest.main()