#!/usr/bin/env python
# coding: utf-8

# **Tensor product spline implementation**

# In[3]:


# convert jupyter notebook to python script
#get_ipython().system('jupyter nbconvert --to script tensorproductspline.ipynb')


# In[1]:


import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from scipy.sparse import kron

from .bspline import B_spline
from .penalty_matrix import PenaltyMatrix


class TensorProductSpline(B_spline, PenaltyMatrix):
    """Implementation of the tensor product spline according to Simon Wood, 2006."""
    
    def __init__(self):
        """It is important that len(x1) == len(x2)."""
        self.x1 = None
        self.x2 = None
        self.basis = None
        
    def tensor_product_spline_2d_basis(self, x_basis=None, k1=5, k2=5, print_shapes=False):
        """Calculate the TPS from two 1d B-splines.
        
        Parameters:
        -------------
        k1 : integer   - Number of knots for the first B-spline.
        k2 : integer   - Number of knots for the second B-Spline.
        print_shape : bool - prints the dimensions of the basis matrices.
        
        """
        
        #print("Use 'x_basis' for the spline basis!")
        self.x1 = x_basis[:,0]
        self.x2 = x_basis[:,1]

        self.x1 = np.unique(self.x1)
        self.x2 = np.unique(self.x2)
        
        self.x1.sort()
        self.x2.sort()
        
        self.k1 = k1
        self.k2 = k2
        BSpline_x1 = BSpline(self.x1)
        BSpline_x2 = BSpline(self.x2)
        BSpline_x1.b_spline_basis(k=self.k1)
        BSpline_x2.b_spline_basis(k=self.k2)
        
        #BSpline_x1.plot_basis("1st B-Spline basis")
        #BSpline_x2.plot_basis("2nd B-Spline basis")
        
        
        self.X1 = BSpline_x1.basis
        self.X2 = BSpline_x2.basis
        #self.basis = kron(self.X1, self.X2).toarray()

        # kronecker product for TPS according to
        # https://stats.stackexchange.com/questions/254542/surface-fit-using-tensor-product-of-b-splines 
        X = np.zeros((self.x1.shape[0], self.X1.shape[1]*self.X2.shape[1]))
        for i in range(X.shape[0]):
            X[i,:] = np.kron(self.X1[i,:], self.X2[i,:])
        self.basis = X
        
        
        if print_shapes:
            print("Shape of the first basis: ", self.X1.shape)
            print("Shape of the second basis: ", self.X2.shape)
            print("Shape of the tensor product basis: ", self.basis.shape)
        return
        
    # not trusted
    def plot_basis(self):
        """Plot the tensor product spline basis matrix for a 2d TPS.
        
        TODO:
            [ ] rework
        """
        fig = go.Figure()
        x1g, x2g = np.meshgrid(self.x1, self.x2)
        #print("x1g: ", x1g.shape)
        #print("x2g: ", x2g.shape)
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
        return

    
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


# In[ ]:


# Test for tensorproductspline
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error


# data generation on a grid
samples = 250
x = np.linspace(-3, 3, samples)
xgrid = np.meshgrid(x,x)
grid = np.array([xgrid[1].ravel(), xgrid[0].ravel()]).T

# define test functions
def binorm(x,y):
    return (15/(2*np.pi)) * np.exp(-0.5 * (x**2 + y**2)) + 0.1*np.random.randn(x.shape[0])

def xtimesy(x,y):
    return x * y  + np.random.randn(1)

Y = binorm(grid[:,0], grid[:,1]).reshape((samples, samples))
#Y = xtimesy(grid[:,0], grid[:,1]).reshape((samples, samples))

go.Figure(go.Surface(z=Y, name="Test function")).show()

T = TensorProductSpline(x1=np.unique(grid[:,0]), 
                        x2=np.unique(grid[:,1]))

T.tensor_product_spline_2d_basis(k1=12, k2=12)

# OLS
X = T.basis
beta = (np.linalg.pinv(X.T @ X) @ X.T @ Y)
# prediction
pred = X @ beta

fig = go.Figure()
fig = make_subplots(rows=1, cols=2,
                    specs=[[{"type": "scene"}, {"type": "scene"}]],)
fig.add_trace(go.Scatter3d(x=grid[:,0], y=grid[:,1], z=Y.ravel(), name="data", mode="markers"), row=1, col=1)
fig.add_trace(go.Scatter3d(x=grid[:,0], y=grid[:,1], z=pred.ravel(), name="pred"), row=1, col=1)
fig.show()

mean_squared_error(pred.ravel(), Y.ravel())
"""

