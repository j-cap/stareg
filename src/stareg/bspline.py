#!/usr/bin/env python
# coding: utf-8

# In[3]:


# convert jupyter notebook to python script
#get_ipython().system('jupyter nbconvert --to script bspline.ipynb')


# In[20]:


import numpy as np
import plotly.graph_objects as go

from .Code_snippets import addVertLinePlotly
from .penalty_matrix import PenaltyMatrix

class B_spline(PenaltyMatrix):
    
    def __init__(self, x=None, order="cubic"):
        self.order = order
        self.x = x
        self.basis = None
        self.knots = None
        if order is "cubic":
            self.m = 2

    def b_spline(self, k, i, m=2):
        """Compute the i-th b-spline basis function of order m at the values given in x.
        
        Parameter:
        ---------------------
        k   : array,     of knot locations
        i   : int,       index of the b-spline basis function to compute
        m   : int,       order of the spline, default is 2 (cubic)
        """
        if m==-1:
            # return 1 if x is in {k[i], k[i+1]}, otherwise 0
            return (self.x >= k[i]) & (self.x < k[i+1]).astype(int)
        else:
            #print("m = ", m, "\t i = ", i)
            z0 = (self.x - k[i]) / (k[i+m+1] - k[i])
            z1 = (k[i+m+2] - self.x) / (k[i+m+2] - k[i+1])
            return z0*self.b_spline(k, i, m-1) + z1*self.b_spline(k, i+1, m-1)
                
    def b_spline_basis(self, x_basis=None, k=10, m=2):
        """Set up model matrix for the B-spline basis.
        One needs k + m + 1 knots for a spline basis of order m with k parameters. 
        If self.x is defined, x_basis is not used!
        
        Parameters:
        -------------
        k : integer   - number of parameters (== number of B-splines)
        m : interger  - specifies the order of the spline, m+1 = order
        x_basis : None, ndarray - for the case that no x was defined
                                  in the initialization of the BSpline
        
        """
        # check_if_none(self.x, x_basis, self)
        if not hasattr(self, 'm'):
            self.m = m
        if not hasattr(self, 'x'):
            self.x = None
        if self.x is None :
            if type(x_basis) is None:
                print("Type of x: ", type(x_basis))
                print("Include data for 'x'!")
                return    
            elif type(x_basis) is np.ndarray:
                print("Use 'x_basis' for the spline basis!")
                self.x = x_basis
            else:
                print(f"Datatype for 'x':{type(x_basis)} not supported!")
                return
        else:
            print("'x' from initialization is used for the spline basis!")
        self.x.sort()
        x = self.x
        assert (type(x) is np.ndarray), "Type of x is not ndarray!"
        n = len(x) # n = number of data
        X = np.zeros((n, k))
        
        xmin, xmax = np.min(x), np.max(x)
        xk = np.quantile(a=x, q=np.linspace(0,1,k - m))
        dx = xk[-1] - xk[-2]
        xk = np.insert(xk, 0, np.arange(xmin-(m+1)*dx, xmin, dx))    
        xk = np.append(xk, np.arange(xmax+dx, xmax+(m+3)*dx, dx))
        
        for i in range(k):
            X[:,i] = self.b_spline(k=xk, i=i, m=m)
            
        self.basis = X
        self.knots = xk
        self.n_param = int(X.shape[1])
        return 
    
    def plot_basis(self, title=""):
        """Plot the B-spline basis matrix and the knot loactions.
        They are indicated by a vertical line.
        """
        if self.basis is None or self.knots is None:
            k = 10
            self.b_spline_basis(k=k, m=self.m)

        fig = go.Figure()
        for i in range(self.basis.shape[1]):
            fig.add_trace(go.Scatter(x=self.x, y=self.basis[:,i], 
                                     name=f"BSpline {i+1}", mode="lines"))
        for i in self.knots:
            addVertLinePlotly(fig, x0=i)
        if title:
            fig.update_layout(title=title)
        else:
            fig.update_layout(title="B-Spline basis")
        fig.show()
        return
                    
