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
        """It is important that len(x1) == len(x2)."""
        self.x1 = None
        self.x2 = None
        self.basis = None
        
    def tensor_product_spline_2d_basis(self, x_data=None, k1=5, k2=5, print_shapes=False, type_="quantile"):
        """Calculate the TPS from two 1d B-splines.
        
        Parameters:
        -------------
        k1 : integer   - Number of knots for the first B-spline.
        k2 : integer   - Number of knots for the second B-Spline.
        print_shape : bool  - prints the dimensions of the basis matrices.
        type_ : int         - "quantile" or "equidistant", describes the knot placement
        
        """
        
        self.x1 = x_data[:,0]
        self.x2 = x_data[:,1]

        self.x1 = np.unique(self.x1)
        self.x2 = np.unique(self.x2)
        
        self.x1.sort()
        self.x2.sort()
        
        self.k1 = k1
        self.k2 = k2
        bspline_x1 = Bspline()
        bspline_x2 = Bspline()
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
        
        
        if print_shapes:
            print("Shape of the first basis: ", self.basis_x1.shape)
            print("Shape of the second basis: ", self.basis_x2.shape)
            print("Shape of the tensor product basis: ", self.basis.shape)
        
    # not trusted
    def plot_basis(self):
        """Plot the tensor product spline basis matrix for a 2d TPS.
        
        TODO:
            [ ] rework
        """
        fig = go.Figure()
        x1g, x2g = np.meshgrid(self.x1, self.x2)

        for i in range(self.basis.shape[1]):
            fig.add_trace(
                go.Surface(
                    x=x1g, y=x2g,
                    z=self.basis[:,i].reshape((self.x1.shape[0], self.x2.shape[0])),
                    name=f"TPS Basis {i+1}",
                    showscale=False
                )
            )
                
        fig.update_layout(
            scene=dict(
                xaxis_title="x1",
                yaxis_title="x2",
                zaxis_title=""
            ),
            title="Tensor product spline basis", 
        )
        fig.show()

    
    # not trusted
    def plot_basis_individuel(self):
        """
        TODO:
            [ ] rework
        """
        dim = self.basis.shape
        dim_resh_1 = int(np.sqrt(dim[0]))
        dim_resh_2 = int(np.sqrt(dim[1]))

        
        fig = make_subplots(rows=dim_resh_2, cols=dim_resh_2)

        for i in range(dim[1]):
            data = self.basis[:,i].reshape((dim_resh_1, dim_resh_1))
            if i < dim_resh_2:
                fig.add_trace(go.Heatmap(z=data), row=1, col=i+1)
            elif i >= dim_resh_2 and i < 2*dim_resh_2:
                fig.add_trace(go.Heatmap(z=data), row=2, col=i+1-dim_resh_2)
            elif i >= 2*dim_resh_2 and i < 3*dim_resh_2:
                fig.add_trace(go.Heatmap(z=data), row=3, col=i+1-2*dim_resh_2)
            elif i >= 3*dim_resh_2 and i < 4*dim_resh_2:
                fig.add_trace(go.Heatmap(z=data), row=4, col=i+1-3*dim_resh_2)
            elif i >= 4*dim_resh_2:
                fig.add_trace(go.Heatmap(z=data), row=5, col=i+1-4*dim_resh_2)

        fig.update_traces(showscale=False)
        fig.show()

