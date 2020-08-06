#!/usr/bin/env python
# coding: utf-8

# In[6]:


# convert jupyter notebook to python script
#get_ipython().system('jupyter nbconvert --to script smooth.ipynb')


# In[1]:


import numpy as np

from .bspline import Bspline
from .tensorproductspline import TensorProductSpline

class Smooths(Bspline):
    """Implementation of the 1d smooth used in Structured Additive Models."""

    def __init__(self, x_data, n_param, constraint="smooth", y_peak_or_valley=None, 
                lambdas=None, type_="quantile"):
        """Create the B-spline basis as well as the constraint matrices for the constraint.
        
        Parameters
        ----------
        x_data : array 
            Data of shape (n_samples, ) to build the B-spline basis for.
        n_param : int
            Number of B-Splines to use for the basis.
        constraint : str
            Type of constraint, one of {"smooth", "inc", "dec", "conv", "conc", "peak", "valley"}.
        y_peak_or_valley : array 
            Response variable of shape (n_samples, ) values to search for the peak or valley, respectively. 
        lambdas : dict
            Smoothing parameter value for the smoothnes and constraint penalty, e.g. {"smoothness": 1, "constraint": 1000}.
        type_   : str
            Describes the knot placement, either "quantile" or "equidistant".

        """
        self.x_data = x_data
        self.n_param = n_param
        self.constraint = constraint
        if lambdas is None:
            self.lam = {"smoothness":1, "constraint": 1000}
        else:
            assert (type(lambdas) == dict), "Need to be of the form {'smoothness':1, 'constraint':1}"
            self.lam = lambdas
        self.knot_type = type_
        self.bspline_basis(x_data=self.x_data, k=self.n_param, type_=type_)
        # Create the penalty matrix for the given penalty
        if constraint == "inc":
            self.penalty_matrix = self.d1_difference_matrix(n_param=self.n_param)
        elif constraint == "dec":
            self.penalty_matrix = -1 * self.d1_difference_matrix(n_param=self.n_param) 
        elif constraint == "conv":
            self.penalty_matrix = self.d2_difference_matrix(n_param=self.n_param)
        elif constraint == "conc":
            self.penalty_matrix = -1 * self.d2_difference_matrix(n_param=self.n_param)
        elif constraint == "smooth":
            self.penalty_matrix = self.smoothness_matrix(n_param=self.n_param)
        elif constraint == "peak":
            assert (y_peak_or_valley is not None), "Include real y_data in Smooths()"
            self.penalty_matrix = self.peak_matrix(
                n_param=self.n_param, basis=self.basis, y_data=y_peak_or_valley
            )
        elif constraint == "valley":
            assert (y_peak_or_valley is not None), "Include real y_data in Smooths()"
            self.penalty_matrix = self.valley_matrix(
                n_param=self.n_param, basis=self.basis, y_data=y_peak_or_valley
            )
        elif constraint == "multi-peak":
            assert (y_peak_or_valley is not None), "Include real y_data in Smooths()"
            self.penalty_matrix = self.multi_peak_matrix(
                n_param=self.n_param, basis=self.basis, y_data=y_peak_or_valley
            )
        elif constraint == "multi-valley":
            assert (y_peak_or_valley is not None), "Include real y_data in Smooths()"
            self.penalty_matrix = self.multi_valley_matrix(
                n_param=self.n_param, basis=self.basis, y_data=y_peak_or_valley
            )
        else:
            print(f"Penalty {constraint} not implemented!")

    
class TensorProductSmooths(TensorProductSpline):
    """Implementation of the 2d tensor product spline smooth in Structured Additive Models."""
    
    def __init__(self, x_data=None, n_param=(1,1), constraint="smooth", lambdas=None, type_="quantile"):
        """Create the tensor product spline basis as well as the smoothness penalty matrices.
        
        Parameters
        ----------
        x_data : array 
            Data of shape (n_samples, 2) to build the TP-spline basis for.
        n_param : tuple
            Number of B-Splines to use for each basis.
        constraint : str
            Type of constraint, one of {"smooth", "inc", "dec", "conv", "conc", "peak", "valley"}.
        lambdas : dict
            Smoothing parameter value for the smoothnes and constraint penalty, e.g. {"smoothness": 1, "constraint": 1000}.
        type_   : str
            Describes the knot placement, either "quantile" or "equidistant".
        
        """

        # TODO:
        # - [ ] constraints need to be implemented

        self.x_data = x_data
        self.x1, self.x2 = x_data[:,0], x_data[:,1]
        self.n_param = n_param
        self.constraint = constraint
        if lambdas is None:
            self.lam = {"smoothness":0, "constraint": 0}
        else:
            self.lam = lambdas
        self.knot_type = type_
        self.tensor_product_spline_2d_basis(x_data=self.x_data, k1=n_param[0], k2=n_param[1], type_=type_)
        # Create the penalty matrix for the given penalty
        if constraint == "smooth":
            print("--- NOT FINISHED ---")
            self.penalty_matrix = np.zeros((np.prod(n_param)-2, np.prod(n_param)))
        else:
            print("--- NOT FINISHED ---")
            print(f"Penalty {constraint} not implemented!")
        

