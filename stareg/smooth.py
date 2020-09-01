#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.linalg import sqrtm

from .bspline import Bspline
from .tensorproductspline import TensorProductSpline

class Smooths(Bspline):
    """Implementation of the 1d smooth used in Structured Additive Models."""

    def __init__(self, x_data, n_param, constraint="smooth", y=None, 
                lambdas=None, type_="quantile"):
        """Create the B-spline basis as well as the constraint matrices for the constraint.
        
        Parameters
        ----------
        x_data : array 
            Data of shape (n_samples, ) to build the B-spline basis for.
        n_param : int
            Number of B-Splines to use for the basis.
        constraint : str
            Type of constraint, one of {inc", "dec", "conv", "conc", "peak", "valley"}.
        y : array 
            Response variable of shape (n_samples, ) values to search for the peak or valley, respectively. 
        lambdas : dict
            Smoothing parameter value for the smoothnes and constraint penalty, e.g. {"smoothness": 1, "constraint": 1000}.
        type_   : str
            Describes the knot placement, either "quantile" or "equidistant".

        """
        self.x_data = x_data
        self.n_param = n_param
        self.coef_ = None
        self.constraint = constraint
        if lambdas is None:
            self.lam = {"smoothness":1, "constraint": 1000}
        else:
            assert (type(lambdas) == dict), "Need to be of the form {'smoothness':1, 'constraint':1}"
            self.lam = lambdas
        self.knot_type = type_
        self.bspline_basis(x_data=self.x_data, k=self.n_param, type_=type_)
        self.smoothness = self.smoothness_matrix(n_param=self.n_param).T @ self.smoothness_matrix(n_param=self.n_param)
        # Create the penalty matrix for the given penalty
        if constraint == "inc":
            self.penalty_matrix = self.d1_difference_matrix(n_param=self.n_param)
        elif constraint == "dec":
            self.penalty_matrix = -1 * self.d1_difference_matrix(n_param=self.n_param) 
        elif constraint == "conv":
            self.penalty_matrix = self.d2_difference_matrix(n_param=self.n_param)
        elif constraint == "conc":
            self.penalty_matrix = -1 * self.d2_difference_matrix(n_param=self.n_param)
        elif constraint == "peak":
            assert (y is not None), self.msg_include_ydata
            self.penalty_matrix = self.peak_matrix(
                n_param=self.n_param, basis=self.basis, y_data=y
            )
        elif constraint == "valley":
            assert (y is not None), self.msg_include_ydata
            self.penalty_matrix = self.valley_matrix(
                n_param=self.n_param, basis=self.basis, y_data=y
            )
        elif constraint == "multi-peak":
            assert (y is not None), self.msg_include_ydata
            self.penalty_matrix = self.multi_peak_matrix(
                n_param=self.n_param, basis=self.basis, y_data=y
            )
        elif constraint == "multi-valley":
            assert (y is not None), self.msg_include_ydata
            self.penalty_matrix = self.multi_valley_matrix(
                n_param=self.n_param, basis=self.basis, y_data=y
            )
        elif constraint == "peak-and-valley":
            assert (y is not None), self.msg_include_ydata
            self.penalty_matrix = self.multi_extremum_matrix(
                n_param=self.n_param, basis=self.basis, y_data=y
            )
        elif constraint == "NoConstraint":
            self.penalty_matrix = np.zeros((self.n_param, self.n_param))
        else:
            print(f"Penalty {constraint} not implemented!")

    
class TensorProductSmooths(TensorProductSpline):
    """Implementation of the 2d tensor product spline smooth in Structured Additive Models."""
    
    def __init__(self, x_data=None, n_param=(1,1), constraint="none", lambdas=None, y=None, type_="quantile"):
        """Create the tensor product spline basis as well as the smoothness penalty matrices.
        
        Parameters
        ----------
        x_data : array 
            Data of shape (n_samples, 2) to build the TP-spline basis for.
        n_param : tuple
            Number of B-Splines to use for each basis.
        constraint : str
            Type of constraint, one of {"none", "inc", "inc_1", "inc_2"}.
        lambdas : dict
            Smoothing parameter value for the smoothnes and constraint penalty, e.g. {"smoothness": 1, "constraint": 1000}.
        y : array 
            Response variable of shape (n_samples, ) values to search for the peak or valley, respectively. 
        type_   : str
            Describes the knot placement, either "quantile" or "equidistant".
        
        """

        # TODO:
        # - [ ] constraints need to be implemented

        self.x_data = x_data
        self.x1, self.x2 = x_data[:,0], x_data[:,1]
        self.n_param = n_param
        self.coef_ = None
        self.constraint = constraint
        if lambdas is None:
            self.lam = {"smoothness":0, "constraint": 0}
        else:
            self.lam = lambdas
        self.knot_type = type_
        self.tensor_product_spline_2d_basis(x_data=self.x_data, k1=n_param[0], k2=n_param[1], type_=type_)
        # Create the penalty matrix for the given penalty

        # create smoothness matrix according to Fahrmeir, p. 508
        K1 = self.smoothness_matrix(n_param=self.n_param[0]).T @ self.smoothness_matrix(n_param=self.n_param[0])
        K2 = self.smoothness_matrix(n_param=self.n_param[1]).T @ self.smoothness_matrix(n_param=self.n_param[1])  
        self.smoothness = np.kron(np.eye(self.n_param[1]), K1) + np.kron(K2, np.eye(self.n_param[0]))

        # TODO: implement the constraints
        if constraint == "none":
            self.penalty_matrix = np.zeros((np.prod(self.n_param), np.prod(self.n_param)))
        elif constraint == "inc":
            #  according to Fahrmeir, p. 508
            P1 = self.d1_difference_matrix(n_param=self.n_param[0])
            P2 = self.d1_difference_matrix(n_param=self.n_param[1])
            K1, K2 = P1.T @ P1, P2.T @ P2
            K = np.kron(np.eye(self.n_param[1]), K1) + np.kron(K2, np.eye(self.n_param[0]))
            self.penalty_matrix = sqrtm(K)
        elif constraint == "inc_1":
            # increasing constraint in dimension 1
            P2 = self.d1_difference_matrix(n_param=self.n_param[1])
            I1, K2 = np.eye(self.n_param[0]), P2.T @ P2
            K = np.kron(K2, I1)
            self.penalty_matrix = sqrtm(K)
        elif constraint == "inc_2":
            # increasing constraint in dimension 2
            P1 = self.d1_difference_matrix(n_param=self.n_param[0])
            I2, K1 = np.eye(self.n_param[1]), P1.T @ P1
            K = np.kron(I2, K1)
            self.penalty_matrix = sqrtm(K)
        elif constraint == "peak":
            #  use basic scheme according to Fahrmeir, p. 508
            peak1 = self.peak_matrix(n_param=self.n_param[0], y_data=y, basis=self.basis_x1)
            peak2 = self.peak_matrix(n_param=self.n_param[1], y_data=y, basis=self.basis_x2)
            K1, K2 = peak1.T @ peak1, peak2.T @ peak2
            K = np.kron(np.eye(self.n_param[1]), K1) + np.kron(K2, np.eye(self.n_param[0]))
            self.penalty_matrix = sqrtm(K)
        else:
            self.penalty_matrix = np.zeros((np.prod(self.n_param), np.prod(self.n_param)))
            print("--- Constraint NOT FINISHED ---")
            print(f"Penalty {constraint} not implemented!")
        

