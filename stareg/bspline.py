#!/usr/bin/env python
# coding: utf-8

import numpy as np
import plotly.graph_objects as go
import copy

from .utils import add_vert_line
from .penalty_matrix import PenaltyMatrix

class Bspline(PenaltyMatrix):
    """Implementation of B-splines according to de Boor, 1978."""

    def __init__(self, order="cubic"):
        """Initialization.

        Parameters
        ----------
        order : str
            Order of the spline, default is 'cubic'.

        """
    
        self.order = order
        self.basis = None
        self.knots = None
        if order == "cubic":
            self.m = 2

    def bspline(self, x, knots, i, m=2):
        """Compute the i-th b-spline basis function of order m at the values given in x.
        
        Note
        ---
        Not intended to be run as standalone function.
        
        Parameters
        ----------
        x : array
            Data to calculate the spline values for.
        k : array
            Array of knot locations.
        i : int
            Index of the b-spline basis function to compute.
        m : int
            Order of the spline, default is 2.
        
        """
        if m==-1:
            # return 1 if x is in {k[i], k[i+1]}, otherwise 0
            return (x >= knots[i]) & (x < knots[i+1]).astype(int)
        else:
            #print("m = ", m, "\t i = ", i)
            z0 = (x - knots[i]) / (knots[i+m+1] - knots[i])
            z1 = (knots[i+m+2] - x) / (knots[i+m+2] - knots[i+1])
            return z0*self.bspline(x, knots, i, m-1) + z1*self.bspline(x, knots, i+1, m-1)
                
    def bspline_basis(self, x_data=None, k=10, m=2, type_="quantile"):
        """Set up model matrix for the B-spline basis.

        Note
        ---
        One needs k + m + 1 knots for a spline basis of order m with k parameters. 
        If self.x is defined, x_data is not used!
        
        Parameters
        ----------
        k : int
            Number of parameters (== number of B-splines).
        m : int
            Specifies the order of the spline, m+1 = order.
        x_data : np.ndarray
            Data of shape (n_samples, ) to compute the B-spline with.
        type_ : str
            Describes the knot placement, either 'quantile' or 'equidistant'.
        
        """

        if not hasattr(self, 'm'):
            self.m = m
        assert (type(x_data) is np.ndarray), "Type of x is not ndarray!"
        
        n = len(x_data) # n = number of data
        X = np.zeros((n, k))
        
        xmin, xmax = np.min(x_data), np.max(x_data)
        if type_ == "quantile":
            xk = np.quantile(a=x_data, q=np.linspace(0,1,k - m))
        elif type_ == "equidistant":
            xk = np.linspace(xmin, xmax, k-m)
        else:
            print("Knot placement type is not supported!!!")
            print("Either 'quantile' or 'equidistant'!")

        dx = np.min(np.diff(xk))
        xk = np.insert(xk, 0, np.linspace(xmin-(m+1)*dx, xmin, 3, endpoint=False))
        xk = np.append(xk, np.linspace(xmax+dx, xmax+(m+1)*dx, 3, endpoint=True))
        
        for i in range(k):
            X[:,i] = self.bspline(x=x_data, knots=xk, i=i, m=m)
            
        self.x = x_data
        self.basis = X
        self.knots = xk
        
        self.knot_type = type_
        self.n_param = int(X.shape[1])
    
    def spp(self, sp=0, coef_=None):
        """Calculate the single point prediction for B-splines given the coefficients. 

        Parameters
        ----------
        sp : float
            Single point to calculate the prediction for.
        coef_ : np.array
            Calculated coefficients for the B-splines.

        Returns
        -------
        p : np.float
            Predicted value. 

        """
        if sp != 1:
            idx = np.argwhere(self.knots > sp)[0][0]
        else:
            idx = np.argwhere(self.knots >= sp)[0][0]
        s = []
        [s.append(self.bspline(x=float(sp), knots=self.knots[idx-4:idx+4], i=i, m=2)) for i in range(4)]
        return sum(s * coef_[idx-4:idx])            

    def plot_basis(self, title=""):
        """Plot the B-spline basis matrix and the knot loactions.

        Parameters
        ----------
        title : str
            Title on the figure. 

        """
        # TODO:
        # - [ ] rework this function
        
        assert (self.basis is not None), "Run .bspline_basis() first!"

        fig = go.Figure()
        for i in range(self.basis.shape[1]):
            fig.add_trace(go.Scatter(x=self.x, y=self.basis[:,i], 
                                     name=f"BSpline {i+1}", mode="lines"))
        for i in self.knots:
            add_vert_line(fig, x0=i)
        if title:
            fig.update_layout(title=title)
        else:
            fig.update_layout(title="B-Spline basis")
        return fig
