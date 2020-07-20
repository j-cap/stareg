#!/usr/bin/env python
# coding: utf-8

# **Implementation of the 1D smooth for a BSpline basis with penalties**

# In[6]:


# convert jupyter notebook to python script
#get_ipython().system('jupyter nbconvert --to script smooth.ipynb')


# In[1]:


import numpy as np

from .bspline import B_spline
from .tensorproductspline import TensorProductSpline

class Smooths(B_spline):
    """Implementation of the 1d smooth used in Additive Models."""

    def __init__(self, x_data, n_param, penalty="smooth", y_peak_or_valley=None, lam_c=None, lam_s=None):
        """Create the B-spline basis as well as the penalty matrices for the penalty.
        
        Parameters:
        -------------------
        x_data  : array of shape (len(x_data), )          - Values to build the B-spline basis for.
        n_param : int                                     - Number of B-Splines to use for the basis.
        penalty : string                                  - Type of penalty, one of "smooth", "inc", "dec"
                                                            "conv", "conc", "peak", "valley".
        y_peak_or_valley : array of shape (len(x_data), ) - Response variable values to search for the peak
                                                            or valley, respectively. 
        lam_c   : float                                   - Smoothing parameter value for the constraint. 
        lam_s   : float                                   - Smoothing parameter value for the smoothness
                                                            penalty.
        -------------------
        """
        self.x_data = x_data
        self.n_param = n_param
        self.penalty = penalty
        self.lam_constraint = lam_c
        self.lam_smooth = lam_s
        self.bspline = B_spline()
        self.b_spline_basis(x_basis=self.x_data, k=self.n_param)
        
        # Sanity check for peak/valley penalty
        if penalty is "peak" or penalty is "valley":
            assert (y_peak_or_valley is not None), "Include real y_data in Smooths()"
        
        # Create the penalty matrix for the given penalty
        if penalty is "inc":
            self.penalty_matrix = self.D1_difference_matrix()
        elif penalty is "dec":
            self.penalty_matrix = -1 * self.D1_difference_matrix() 
        elif penalty is "conv":
            self.penalty_matrix = self.D2_difference_matrix()
        elif penalty is "conc":
            self.penalty_matrix = -1 * self.D2_difference_matrix()
        elif penalty is "smooth":
            self.penalty_matrix = self.Smoothness_matrix()
        elif penalty is "peak":
            self.penalty_matrix = self.Peak_matrix(basis=self.basis, y_data=y_peak_or_valley)
        elif penalty is "valley":
            self.penalty_matrix = self.Valley_matrix(basis=self.basis, y_data=y_peak_or_valley)
        else:
            print(f"Penalty {penalty} not implemented!")
    
class TensorProductSmooths(TensorProductSpline):
    """Implementation of the 2d tensor product spline smooth in Additive Models."""
    
    def __init__(self, x_data=None, n_param=(1,1), penalty="smooth", lam_c=None, lam_s=None):
        """Create the tensor product spline basis as well as the smoothness penalty matrices.
        
        Parameters:
        -------------------
        x_data  : array of shape (len(x_data), 2)         - Values to build the B-spline basis for.
        n_param : tuple of integer                        - Number of B-Splines per dimension.
        penalty : string                                  - Type of penalty, currently only "smooth"
        lam_c   : float                                   - Smoothing parameter value for the constraint. 
        lam_s   : float                                   - Smoothing parameter value for the smoothness
                                                            penalty.
        -------------------
        """
        
        self.x_data = x_data
        self.x1, self.x2 = x_data[:,0], x_data[:,1]
        self.n_param = n_param
        self.penalty = penalty
        self.lam_constraint = lam_c
        self.lam_smooth = lam_s
        self.tps = TensorProductSpline()
        self.tensor_product_spline_2d_basis(x_basis=self.x_data, k1=n_param[0], k2=n_param[1])
        
        # Create the penalty matrix for the given penalty
        if penalty is "smooth":
            print(f"Penalty [{penalty}] Needs to be implemented!")
            #self.penalty_matrix = self.Smoothness_matrix()
        else:
            print(f"Penalty {penalty} not implemented!")
        


# In[14]:




