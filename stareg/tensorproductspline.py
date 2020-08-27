#!/usr/bin/env python
# coding: utf-8

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.sparse import kron
from .bspline import Bspline
from .penalty_matrix import PenaltyMatrix


class TensorProductSpline(Bspline, PenaltyMatrix):
    """Implementation of the tensor product spline according to Simon Wood, 2006."""
    
    def __init__(self):
        """Initialization."""
        pass
        
    def tensor_product_spline_2d_basis(self, x_data=None, k1=5, k2=5, type_="quantile"):
        """Calculate the tensor product spline basis from two 1d B-splines.
        
        Parameters
        ----------
        x_data : np.ndarray
            Data of the shape (n_samples, 2).
        k1 : int
            Number of knots for the first B-spline.
        k2 : int
            Number of knots for the second B-Spline.
        type_ : str 
            Describes the knot placement, either "quantile" or "equidistant".    
        
        """
        
        self.x1, self.x2 = x_data[:,0], x_data[:,1]
        self.k1, self.k2 = k1, k2
        bspline_x1, bspline_x2 = Bspline(), Bspline()
        bspline_x1.bspline_basis(x_data=self.x1, k=self.k1, type_=type_)
        bspline_x2.bspline_basis(x_data=self.x2, k=self.k2, type_=type_)               
        self.basis_x1 = bspline_x1.basis
        self.basis_x2 = bspline_x2.basis
        # kronecker product for TPS according to
        # https://stats.stackexchange.com/questions/254542/surface-fit-using-tensor-product-of-b-splines 
        X = np.zeros((self.x1.shape[0], self.basis_x1.shape[1]*self.basis_x2.shape[1]))
        for i in range(X.shape[0]):
            X[i,:] = np.kron(self.basis_x1[i,:], self.basis_x2[i,:])

        self.basis = X
        self.knots_x1 = bspline_x1.knots 
        self.knots_x2 = bspline_x2.knots 


    def spp(self, sp=0, coef_=None):
        """Calculate the single point prediction for TP-splines given the coefficients. 

        Parameters
        ----------
        sp : float
            Single point to calculate the prediction for, shape (1, 2)
        coef_ : np.array
            Calculated coefficients for the TP-splines.
        knots : np.array
            Knot sequence. 

        Returns
        -------
        p : np.float
            Predicted value. 

        """

        s1, s2 = [], []
        for i in range(int(np.sqrt(len(coef_)))):
            s1.append(self.bspline(x=sp[0], knots=self.knots_x1, i=i, m=2))
            s2.append(self.bspline(x=sp[1], knots=self.knots_x2, i=i, m=2))
        p = np.kron(s1, s2) @ coef_
        return p