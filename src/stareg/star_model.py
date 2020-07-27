#!/usr/bin/env python
# coding: utf-8

# In[2]:


# convert jupyter notebook to python script
#get_ipython().system('jupyter nbconvert --to script star_model.ipynb')


# In[24]:


import plotly.graph_objects as go
import numpy as np
import pandas as pd
import pprint
import copy
from numpy.linalg import lstsq
from scipy.linalg import block_diag
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y 
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import ParameterGrid

from .smooth import Smooths as s
from .smooth import TensorProductSmooths as tps
from .tensorproductspline import TensorProductSpline as t
from .penalty_matrix import PenaltyMatrix

from .utils import check_constraint, check_constraint_full_model


class StarModel(BaseEstimator):
    
    def __init__(self, descr):
        """
        descr : tuple - ever entry describens one part of 
                        the model, e.g.
                        descr =( ("s(1)", "smooth", 25, (1, 100), "equidistant"),
                                 ("s(2)", "inc", 25, (1, 100), "quantile"), 
                                 ("t(1,2)", "smooth", [5,5], (1, 100), "quantile"), 
                               )
                        with the scheme: (type of smooth, type of constraint, 
                                          number of knots, lambdas),
               
        TODO:
            [x] incorporate tensor product splines
        """
        self.description_str = descr
        self.description_dict = {
            t: {"constraint": p, "n_param": n, 
                "lam" : {"smoothness": l[0], "constraint": l[1]},
                "knot_type": k
               } 
            for t, p, n, l, k  in self.description_str}
        self.smooths = None
        self.coef_ = None
        
    def __str__(self):
        pp = pprint.PrettyPrinter()
        return pp.pformat(self.description_dict)

    
    def create_basis(self, X, y=None):
        """Create the unpenalized BSpline basis for the data X.
        
        Parameters:
        ------------
        X : np.ndarray - data
        y : np.ndarray or None  - For peak/valley penalty. 
                                  Catches assertion if None and peak or valley penalty. 
        type_ : str  - "quantile" or "equidistant"  - describes the knot placement
        TODO:
            [x] include TPS
        
        """

        self.smooths = list()
        # generate the smooths according to description_dict       
        for k,v in self.description_dict.items():
            if k[0] == "s":
                self.smooths.append(
                    s(
                        x_data=X[:,int(k[2])-1], 
                        n_param=v["n_param"], 
                        constraint=v["constraint"], 
                        y_peak_or_valley=y,
                        lambdas=v["lam"],
                        type_=v["knot_type"]
                    )
                )
            elif k[0] == "t":
                self.smooths.append(
                    tps(
                        x_data=X[:, [int(k[2])-1, int(k[4])-1]], 
                        n_param=list(v["n_param"]), 
                        constraint=v["constraint"],
                        lambdas=v["lam"],
                        type_=v["knot_type"]
                    )
                )    
        
        self.basis = np.concatenate([smooth.basis for smooth in self.smooths], axis=1) 
        # generate the smootness penalty block matrix
        self.smoothness_penalty_list = [s.lam["smoothness"] * s.smoothness_matrix().T @ s.smoothness_matrix() for s in self.smooths]
        self.smoothness_penalty_matrix = block_diag(*self.smoothness_penalty_list)

        n_coef_list = [0] + [np.product(smooth.n_param) for smooth in self.smooths]
        self.coef_list = np.cumsum(n_coef_list)            
                                                
    def create_constraint_penalty_matrix(self, beta_test=None, y=None):
        """Create the penalty block matrix specified in self.description_str.
        
        Looks like: ----------------------------------- 
                    |lam1*p1 0        0        0      |  
                    |0       lam2*p2  0        0      | = P 
                    |0       0        lam3*p3  0      |
                    |0       0        0        lam4*p4|
                    -----------------------------------
        where P is a a matrix according to the specified penalty. 
        
        The matrix P.T @ P is then used!!!

        Parameters:
        ---------------
        beta_test  : array  - Test beta for sanity checks.
        y          : array  - True y for the peak/valley constraint check
        
        TODO:
            [x]  include the weights !!! 
            [x]  include TPS smoothnes penalty
            [ ]  include TPS shape penalty
        
        """
        assert (self.smooths is not None), "Run Model.create_basis() first!"
        assert (y is not None), "Include y-data for peak/valley constraint!"
        
        if beta_test is None:
            beta_test = np.zeros(self.basis.shape[1])
        
        self.constraint_penalty_list = []
        
        for idx, smooth in enumerate(self.smooths):
            b = beta_test[self.coef_list[idx]:self.coef_list[idx+1]]
            D = smooth.penalty_matrix
            V = check_constraint(beta=b, constraint=smooth.constraint, y=y, basis=smooth.basis)
            self.constraint_penalty_list.append(smooth.lam["constraint"] * D.T @ V @ D )
            
        self.constraint_penalty_matrix = block_diag(*self.constraint_penalty_list)

    def create_basis_for_prediction(self, X=None):
        """Creates unpenalized BSpline basis for the data X.
        
        Parameters:
        X : np.ndarray - prediction data
        """
        self.pred_smooths = list()
        if len(X.shape) == 1:
            X = X.reshape(len(X), -1)

        for k,v in self.description_dict.items():
            if k[0] == "s":
                self.pred_smooths.append(
                    s(x_data=X[:,int(k[2])-1], n_param=v["n_param"], type_=v["knot_type"]))
            elif k[0] == "t":
                self.pred_smooths.append(
                    tps(x_data=X[:, [int(k[2])-1, int(k[4])-1]], n_param=list(v["n_param"]), type_=v["knot_type"]))    
        
        self.basis_for_prediction = np.concatenate([smooth.basis for smooth in self.pred_smooths], axis=1)
    
    def calc_LS_fit(self, X, y):
        """Calculate the basis least squares fit without penalties.

        Parameters:
        --------------
        X : pd.DataFrame or np.ndarray        - data
        y : pd.DataFrame or np.array          - target values
        
        Returns:
        -------------
        self  : object                        - the trained model
        """
        
        self.create_basis(X=X, y=y.ravel())    
        fitting = lstsq(a=self.basis, b=y, rcond=None)
        beta_0 = fitting[0].ravel()
        self.coef_, self.LS_coef_ = beta_0, beta_0        
        return self

    def create_df_for_beta(self, beta_init=None):
        """Craete a dataframe to save all coefficients beta during the fit.

        Parameters:
        ------------
        beta_init  : array          - array of coefficients

        Returns:
        ------------
        df         : pd.DataFrame   - one colume for each coefficient.

        """

        col_name = [ f"b_{i}" for i in range(len(beta_init))]        
        df = pd.DataFrame(columns=col_name)
        d = dict(zip(col_name, beta_init))
        df = df.append(pd.Series(d), ignore_index=True)
        return df

    def fit(self, X, y, plot_=True, max_iter=5):
        """Lstsq fit using Smooths.
        
        Parameters:
        -------------
        X : pd.DataFrame or np.ndarray        - data
        y : pd.DataFrame or np.array          - target values
        max_iter : int                        - maximal iteration of the reweighted LS
        type_ : "quantile" or "equidistant"   - knot placement 

        plot_ : boolean
        
        TODO:
            [x] check constraint violation in the iterative fit
            [x] incorporate TPS in the iterative fit
        """
        
        X, y = check_X_y(X, y)
        self = self.calc_LS_fit(X=X, y=y)
        df = self.create_df_for_beta(beta_init=self.coef_)
        
        for _ in range(max_iter):
            self.create_constraint_penalty_matrix(beta_test=self.coef_, y=y)
            DVD = self.constraint_penalty_matrix
            DD = self.smoothness_penalty_matrix
            BB, By = self.basis.T @ self.basis, self.basis.T @ y
            v_old = check_constraint_full_model(model=self, y=y)
            beta_new = (np.linalg.pinv(BB + DD + DVD) @ By).ravel()
            v_new = check_constraint_full_model(model=self, y=y)
            self.coef_ = beta_new                       
            df = df.append(pd.DataFrame(data=beta_new.reshape(1,-1), columns=df.columns))
            delta_v = np.sum(v_new - v_old)
            if delta_v == 0: 
                break

        print("Iteration converged!")
        print(f"Violated Constraints: {np.sum(check_constraint_full_model(model=self, y=y))} from {len(self.coef_)} ")
        self.df = df
        self.mse = mean_squared_error(y, self.basis @ self.coef_)       
        if plot_: self.plot_fit(X=X, y=y).show()

        return self
    
    def plot_fit(self, X, y):
        """Plot the fitted values with the actual values.

        Parameters:
        --------------
        X   : nd.array      - data
        y   : array         - target values
        y_pred : array      - prediciton

        Returns:
        --------------
        fig : plotly.graph_objs.Figure 

        """
        X, y = check_X_y(X=X, y=y)
        dim = X.shape[1]
        fig = go.Figure()
        if dim == 1:
            fig.add_trace(go.Scatter(x=X[:,0], y=y.ravel(), name="Data", mode="markers"))
            fig.add_trace(go.Scatter(x=X[:,0], y=self.basis @ self.coef_, name="Fit", mode="markers"))
        elif dim == 2:
            fig.add_trace(go.Scatter3d(x=X[:,0], y=X[:,1], z=y.ravel(), name="Data", mode="markers"))
            fig.add_trace(go.Scatter3d(x=X[:,0], y=X[:,1], z=self.basis @ self.coef_, name="Fit", mode="markers"))
            
        fig.update_traces(
            marker=dict(size=8,line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
        fig.update_layout(autosize=True) #, width=500, height=500)
        return fig

    def plot_fit_and_LS_fit(self, X, y):
        """Plot the fitted values with the actual values + the least squares fit without constraint.

        Parameters:
        --------------
        X   : nd.array      - data
        y   : array         - target values
        y_pred : array      - prediciton

        Returns:
        --------------
        fig : plotly.graph_objs.Figure 

        """
        X, y = check_X_y(X=X, y=y)
        fig = self.plot_fit(X=X, y=y)
        fig.add_trace(go.Scatter(
            x=X, y=self.basis @ self.LS_coef_, mode="markers+lines", name="Pure Least Squares Fit")
        )
        return fig       

    def predict(self, X):
        """Prediction of the trained model on the data in X.
        
        Parameters:
        ---------------
        X : array   - Data to predict values for.
        
        Returns:
        ---------------
        pred : array  - Returns the predicted values. 
        """
        
        check_is_fitted(self, attributes="coef_", msg="Estimator is not fitted when using predict()!")

        self.create_basis_for_prediction(X=X)
        return self.basis_for_prediction @ self.coef_
        
    def calc_hat_matrix(self):
        """Calculates the hat or smoother matrix according to
            S = Z (Z'Z + S'S + P'P) Z'

        Returns:
        ------------
        H : np.ndarray    - hat or smoother matrix of size n_data x n_data

        """
        if hasattr(self, "basis") and hasattr(self, "constraint_penalty_matrix") and hasattr(self, "smoothness_penalty_matrix"):
            Z = self.basis
            P = self.constraint_penalty_matrix
            S = self.smoothness_penalty_matrix
            H = Z @ np.linalg.pinv(Z.T @ Z + + S.T @ S + P.T @ P) @ Z.T
        else:
            print("Fit the model to data!")
        return H

    def calc_GCV_score(self, y):
        """Calculate the generalized cross validation score according to Fahrmeir, Regression 2013 p.480
        
        Parameter:
        ------------
        y   : array  - target values

        Returns:
        ------------
        GCV : float  - generalized cross validation score
        
        """
        y_hat = self.basis @ self.coef_
        n = y.shape[0]
        trace = np.trace(self.calc_hat_matrix())
        GCV = (1/n) * np.sum((y.ravel() - y_hat.ravel())**2 / (1 - trace/n))
        return GCV

    def generate_GCV_parameter_list(self, n_grid=5, p_min=1e-4):
        """Generate the exhaustive parameter list for the GCV. 
        
        Parameters:
        --------------
        n_grid  : int               - spacing of the parameter range
        p_min   : float             - minimum parameter value

        Returns:
        --------------
        grid    : ParameterGrid()   - returns an iterator

        """

        params = dict()
        for k in list(self.description_dict):
            for k2, v2 in self.description_dict[k]["lam"].items():
                params[k+"_"+k2] = v2        

        for k,v in params.items():
            spacing = np.geomspace(p_min, p_min*10**n_grid, n_grid, endpoint=False)
            lam_type = k[k.find("_")+1]
            if lam_type == "s":
                params[k] = spacing
            elif lam_type == "c":
                params[k] = spacing * 1e6

        grid = ParameterGrid(params)

        return grid

    def calc_GCV(self, X, y, n_grid=5, p_min=1e-4, plot_=False):
        """Carry out a cross validation for the model.

        Parameters:
        -------------
        X       : nd.array  - data
        y       : array     - target values
        n_grid  : int       - size of the gird per hyperparameter
        p_min   : float     - minimum parameter value
        """
        X, y = check_X_y(X=X, y=y)
        grid = self.generate_GCV_parameter_list(n_grid=n_grid, p_min=p_min)
        # generate dictionary of all hyperparameters (2 per smooth/tps)
        gcv_scores = []
        violated_constraints_list = []

        for idx, params in enumerate(grid):
            for k,v in params.items():
                for k2, v2 in self.description_dict.items():
                    if k[:k.find("_")] == k2:
                        self.description_dict[k2]["lam"][k[k.find("_")+1:]] = v
            #print("\n Parameters: ", params)
            self.fit(X=X, y=y.ravel(), plot_=plot_)
            if plot_:
                print(f"Parameters: {params}")

            ccfm = check_constraint_full_model(model=self, y=y)
            gcv_new = self.calc_GCV_score(y=y)
            
            # add a penalty for violating constraints
            gcv_new += (np.sum(ccfm) / len(self.coef_))
            gcv_scores.append(gcv_new)
            violated_constraints_list.append(ccfm)

        
        gcv_best = list(grid)[np.argmin(gcv_scores)] 
        print(f"Best fit parameter according to adapted-GCV score: {gcv_best}")
        print(f"Violated Constraints: {np.sum(violated_constraints_list[np.argmin(gcv_scores)])} from {len(self.coef_)}")

        print("\n--- BEST FIT ACCORDING TO GCV ---")
        descr_dict = self.set_params_after_gcv(params=gcv_best)
        self.description_dict = descr_dict
        self.fit(X=X, y=y, plot_=True)
        self.gcv_best = gcv_best

        return self
            

    def calc_mse(self, y):
        """Calculates and prints the MSE.
        
        Parameters:
        --------------
        y : array    - target values for training data.
        """
        assert (self.coef_ is not None), "Model is untrained, run Model.fit(X, y) first!"
        y_fit = self.basis @ self.coef_
        mse = mean_squared_error(y, y_fit) 
        print(f"Mean squared error on data: {np.round(mse, 4)}")
        return mse   


    def get_params(self, deep=True):
        """Returns a dict of __init__ parameters. If deep==True, also return 
            parameters of sub-estimators (can be ignored)
        """
        return self.description_dict

    def set_params(self, params):
        """Sets the parameters of the object to the given in params.
        Use self.get_params() as template

        Parameters:
        -----------

        params : dict of dict          : should have the form of self.get_params
        
        """
        for smooth, param  in zip(self.smooths, params.items()):
            for key, value in param[1].items():
                if hasattr(smooth, key):
                    setattr(smooth, str(key), value)

    def set_params_after_gcv(self, params):
        """Sets the parameters after the GCV search. 
        Only run inside of calc_GCV.

        Parameters:
        ------------
        params  : dict      - best parameter combination found by GCV
        """
        descr_dict = copy.deepcopy(self.description_dict)

        for k in params.keys():
            idx = k.find("_")
            s = k[:idx]
            t = k[idx+1:]
            descr_dict[s]["lam"][t] = params[k]
        return descr_dict
         
    
    def plot_xy(self, x, y, title="Titel", name="Data", xlabel="xlabel", ylabel="ylabel"):
        """Basic plotting function."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="markers"))
        fig.update_layout(title=title)
        fig.update_xaxes(title=xlabel)
        fig.update_yaxes(title=ylabel)
        return fig
    
    def plot_1d_basis(self, x, y=None):
        """Plot the basis for 1d fits.
        
        Parameters:
        ------------
        x  : array    - data
        
        """
        assert (self.coef_ is not None), "Fit the model first!"
        fig = go.Figure()
        for i in range(self.basis.shape[1]):
            fig.add_trace(go.Scatter(
                x=x.ravel(), y=self.basis[:,i] * self.coef_[i], mode="lines", name=f"B_{i+1}")
            )
        if y is not None: fig.add_trace(go.Scatter(x=x.ravel(), y=y.ravel(), name="Data", mode="markers"))
#        fig.update_layout(template="plotly_dark")
        fig.show()
                        
 

