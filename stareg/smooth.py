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
        
        ## include the "grid weights" when we use quantile based knot placement 
        #smoothness = self.smoothness_matrix(n_param=self.n_param)
        #if type_ == "quantile":
        #    eff_area = []
        #    for i in range(n_param):
        #        eff_area.append(self.knots[i+4] - self.knots[i])
        #    eff_area = np.array(eff_area)
        #elif type_ == "equidistant":
        #    eff_area = np.ones(n_param) * (self.knots[2]-self.knots[1])
        ## take inverse of the effective area
        #inv_norm_eff_area = (1 / eff_area) # / sum(1 / eff_area)
        # 
        #if np.any(inv_norm_eff_area > 1e5):
        #    print("Inverse of Effective Area is very large! => Downscaling")
        #    inv_norm_eff_area /= 1e5
        #self.inv_eff_area = inv_norm_eff_area
        #smoothness = smoothness * inv_norm_eff_area
        #self.smoothness = smoothness.T @ smoothness

        #print("Try the new difference star approach.")
        # difference star according to Num Strömungsmechanik, Ferziger & Peric
        if type_ == "quantile":
            msv = self.x_data[np.argmax(self.basis, axis=0)]
            msv[0] = self.knots[2]
            msv[-1] = self.knots[-3]
            S = np.zeros((self.n_param-2,self.n_param))
            for i in range(1, S.shape[0]):
                
                s_left = 1 / ((msv[i+1] - msv[i])*(msv[i]-msv[i-1]))
                s_center = (msv[i+1]-msv[i-1])/((msv[i+1]-msv[i])**2 * (msv[i]-msv[i-1]))
                s_right =  1 / (msv[i+1]-msv[i])**2
                S[i, i] = s_left
                S[i, i+1] = s_center
                S[i, i+2] = s_right
            S /= S.max()
            self.smoothness = S.T @ S
        elif type_ == "equidistant":
            self.smoothness = self.smoothness_matrix(n_param=self.n_param)

        # Create the penalty matrix for the given penalty
        if constraint == "none":
            self.penalty_matrix = np.zeros((n_param, n_param))
        elif constraint == "inc":
            self.penalty_matrix = self.d1_difference_matrix(n_param=self.n_param)
        elif constraint == "dec":
            self.penalty_matrix = self.d1_difference_matrix(n_param=self.n_param) 
        elif constraint == "conv":
            self.penalty_matrix = self.d2_difference_matrix(n_param=self.n_param)
        elif constraint == "conc":
            self.penalty_matrix = self.d2_difference_matrix(n_param=self.n_param)
        elif constraint == "peak":
            assert (y is not None), self.msg_include_ydata
            self.penalty_matrix, self.peak_idx = self.peak_matrix(
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
            peak1 = self.peak_matrix(n_param=self.n_param[0], y_data=y, basis=self.basis_x1)[0]
            peak2 = self.peak_matrix(n_param=self.n_param[1], y_data=y, basis=self.basis_x2)[0]
            K1, K2 = peak1.T @ peak1, peak2.T @ peak2
            K = np.kron(np.eye(self.n_param[1]), K1) + np.kron(K2, np.eye(self.n_param[0]))
            self.penalty_matrix = sqrtm(K)
        else:
            self.penalty_matrix = np.zeros((np.prod(self.n_param), np.prod(self.n_param)))
            print("--- Constraint NOT FINISHED ---")
            print(f"Penalty {constraint} not implemented!")
        

