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
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.model_selection import ParameterGrid

from .smooth import Smooths as s
from .smooth import TensorProductSmooths as tps
from .tensorproductspline import TensorProductSpline as t
from .penalty_matrix import PenaltyMatrix

class StarModel(BaseEstimator):
    
    def __init__(self, descr):
        """
        descr : tuple - ever entry describens one part of 
                        the model, e.g.
                        descr =( ("s(1)", "smooth", 25, (1, 100), "equidistant"),
                                 ("s(2)", "inc", 25, (1, 100), "quantile"), 
                                 ("t(1,2)", "tps", [5,5], (1, 100), "quantile"), 
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
        self.y = None
        self.X = None
        
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
        self.y = y
        
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
               
        self.smoothness_penalty_list = [np.sqrt(s.lam["smoothness"]) * s.smoothness_matrix(n_param=s.n_param) for s in self.smooths]
        self.smoothness_penalty_matrix = block_diag(*self.smoothness_penalty_list)

        n_coef_list = [0] + [np.product(smooth.n_param) for smooth in self.smooths]
        n_coef_cumsum = np.cumsum(n_coef_list)
        self.coef_list = n_coef_cumsum            

    
    def create_basis_for_prediction(self, X=None):
        """Creates unpenalized BSpline basis for the data X.
        
        Parameters:
        X : np.ndarray - data
        """
        
        if X is None:
            X = self.X_pred
                
        self.pred_smooths = list()
        for k,v in self.description_dict.items():
            if k[0] == "s":
                self.pred_smooths.append(s(x_data=X[:,int(k[2])-1], n_param=v["n_param"]))
            elif k[0] == "t":
                self.pred_smooths.append(tps(x_data=X[:, [int(k[2])-1, int(k[4])-1]], n_param=list(v["n_param"])))    
        
        self.basis_for_prediction = np.concatenate([smooth.basis for smooth in self.pred_smooths], axis=1)

        X.sort(axis=0)
        del self.X_pred
        self.X_pred = X
        
                                                               
    def create_penalty_block_matrix(self, beta_test=None):
        """Create the penalty block matrix specified in self.description_str.
        
        Looks like: ------------
                    |p1 0  0  0|  
                    |0 p2  0  0|
                    |0  0 p3  0|
                    |0  0  0 p4|
                    ------------
        where p_i is a a matrix according to the specified penalty.

        Parameters:
        ---------------
        beta_test  : array  - Test beta for sanity checks.
        
        TODO:
            [x]  include the weights !!! 
            [x]  include TPS smoothnes penalty
            [ ]  include TPS shape penalty
        
        """
        assert (self.smooths is not None), "Run Model.create_basis() first!"
        assert (self.y is not None), "Run Model.fit(X,y) first!"
        
        if beta_test is None:
            beta_test = np.zeros(self.basis.shape[1])
        
        idx = 0      
        penalty_matrix_list = []
        
        for smooth in self.smooths:
            
            n = smooth.basis.shape[1]
            b = beta_test[idx:idx+n]
            
            D = smooth.penalty_matrix
            V = check_constraint(beta=b, constraint=smooth.constraint, y=self.y, basis=smooth.basis)
            
            penalty_matrix_list.append(np.sqrt(smooth.lam["constraint"]) * D.T @ V @ D )
            idx += n
            
        self.penalty_matrix_list = penalty_matrix_list
        self.penalty_block_matrix = block_diag(*penalty_matrix_list)

    
    def fit(self, X, y, plot_=True, max_iter=5):
        """Lstsq fit using Smooths.
        
        Parameters:
        -------------
        X : pd.DataFrame or np.ndarray
        y : pd.DataFrame or np.array
        max_iter : int          - maximal iteration of the reweighted LS
        type_ : "quantile" or "equidistant"    - knot placement 

        plot_ : boolean
        
        TODO:
            [x] check constraint violation in the iterative fit
            [x] incorporate TPS in the iterative fit
        """
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        self.X, self.y = X, y.ravel()
        # create the basis for the initial fit without penalties
        self.create_basis(X=self.X, y=self.y)    

        fitting = lstsq(a=self.basis, b=y, rcond=None)
        beta_0 = fitting[0].ravel()
        self.coef_ = beta_0
        # self.calc_mse(y=y)
        
        # check constraint violation
        v_old = check_constraint_full_model(self)
        
        # create dataframe to save the beta values 
        col_name = [ f"b_{i}" for i in range(len(beta_0))]        
        df_beta = pd.DataFrame(columns=col_name)
        d = dict(zip(col_name, beta_0))
        df_beta = df_beta.append(pd.Series(d), ignore_index=True)
        
        beta = np.copy(beta_0)
        
        for i in range(max_iter):
            #print("Create basis with penalty and weight")
            self.create_penalty_block_matrix(beta_test=beta)
            
            #print("Least squares fit iteration ", i+1)
            B = self.basis
            d_c = self.penalty_block_matrix
            d_s = self.smoothness_penalty_matrix
        
            bb = B.T @ B
            by = B.T @ y

            # user defined constraint
            dvd = d_c.T             
            # smoothing constraint
            dd = d_s.T @ d_s
            
            beta_new = (np.linalg.pinv(bb + dd + dvd) @ by).ravel()

            self.calc_mse(y=y)
            
            # create dict
            d = dict(zip(col_name, beta_new))
            df_beta = df_beta.append(pd.Series(d), ignore_index=True)
            
            
            # check constraint violation
            v_new = check_constraint_full_model(self)
            
            delta_v = np.sum(v_new - v_old)
            if delta_v == 0:
                print("Iteration converged!")
                break
            else:
                v_old = v_new                
                beta = beta_new
                print("\n Violated constraints: ", np.sum(v_new))
            
        self.df_beta = df_beta
        self.coef_ = self.df_beta.iloc[-1].values
        
        y_fit = self.basis @ self.coef_
     
        self.mse = mean_squared_error(y, y_fit)
        
        if plot_:
            dim = X.shape[1]
            if dim == 1:
                fig = self.plot_xy(x=X[:,0], y=y.ravel(), name="Data")
                fig.add_trace(go.Scatter(x=X[:,0], y=y_fit, name="Fit", mode="markers"))
            elif dim == 2:
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(x=X[:,0], y=X[:,1], z=y.ravel(), name="Data", mode="markers"))
                fig.add_trace(go.Scatter3d(x=X[:,0], y=X[:,1], z=y_fit, name="Fit", mode="markers"))

                
            fig.update_traces(
                marker=dict(
                    size=8, 
                    line=dict(width=2, color='DarkSlateGrey')),
                selector=dict(mode='markers'))

            fig.update_layout(autosize=False, width=500, height=500)
            fig.show()
        
        print(f"Violated Constraints: {np.sum(check_constraint_full_model(self))} from {len(self.coef_)} ")
            
        return y_fit
    
    def predict(self, X):
        """Prediction of the trained model on the data in X.
        
        Parameters:
        ---------------
        X : array   - Data to predict values for.
        
        Returns:
        ---------------
        pred : array  - Returns the predicted values. 
        """
        
        check_is_fitted(self)

        self.X_pred = np.copy(X)
        self.create_basis_for_prediction()
        pred = self.basis_for_prediction @ self.coef_
        
        return pred
    
    def calc_hat_matrix(self):
        """Calculates the hat or smoother matrix according to
            S = Z (Z'Z + P'P) Z'

        Returns:
        ------------
        S : np.ndarray    - hat or smoother matrix of size n_data x n_data

        """
        if hasattr(self, "basis") and hasattr(self, "penalty_block_matrix"):
            Z = self.basis
            P = self.penalty_block_matrix
            S = Z @ np.linalg.pinv(Z.T @ Z + P.T @ P) @ Z.T
        else:
            print("Fit the model to data!")
        return S

    def calc_GCV_score(self):
        """Calculate the generalized cross validation score according to Fahrmeir, Regression 2013 p.480
        
        Returns:
        ------------
        GCV : float  - generalized cross validation score
        
        """
        y = self.y
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
        p_min   : float             - minimum parameter

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

    def calc_GCV(self, n_grid=5, plot_=False):
        """Carry out a cross validation for the model.

        Parameters:
        -------------
        n_grid  : int    - size of the gird per hyperparameter

        """
        grid = self.generate_GCV_parameter_list(n_grid=n_grid)
        # generate dictionary of all hyperparameters (2 per smooth/tps)
        gcv_scores = []
        violated_constraints_list = []

        for idx, params in enumerate(grid):
            for k,v in params.items():
                for k2, v2 in self.description_dict.items():
                    if k[:k.find("_")] == k2:
                        self.description_dict[k2]["lam"][k[k.find("_")+1:]] = v
            #print("\n Parameters: ", params)
            self.fit(X=self.X, y=self.y.ravel(), plot_=plot_)
            if plot_:
                print(f"Parameters: {params}")

            ccfm = check_constraint_full_model(self)
            gcv_new = self.calc_GCV_score()
            
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
        self.fit(X=self.X, y=self.y, plot_=True)

        return gcv_best, descr_dict
            

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
    
    def plot_basis(self, matrix):
        """Plot the matrix."""
        go.Figure(go.Image(z=matrix)).show()
                        
    
def check_constraint(beta, constraint, print_idx=False, y=None, basis=None):
    """Checks if beta fits the constraint.
    
    Parameters:
    ---------------
    beta  : array     - Array of coefficients to be tested against the constraint.
    constraint : str  - Name of the constraint.
    print_idx : bool  - .
    y  : array        - Array of data for constraint "peak" and "valley".
    basis : ndarray   - Matrix of the basis for constraint "peak" and "valley".
    
    Returns:
    ---------------
    V : ndarray       - Matrix with 1 where the constraint is violated, 0 else.
    
    """

    b_diff = np.diff(beta)
    b_diff_diff = np.diff(b_diff)
    if constraint == "inc":
        v = [0 if i > 0 else 1 for i in b_diff] #+ [0]
    elif constraint == "dec":
        v = [0 if i < 0 else 1 for i in b_diff] #+ [0]
    elif constraint == "conv":
        v = [0 if i > 0 else 1 for i in b_diff_diff] #+ [0,0]
    elif constraint == "conc":
        v = [0 if i < 0 else 1 for i in b_diff_diff] #+ [0,0]
    elif constraint == "no":
        v = list(np.zeros(len(beta), dtype=np.int))
    elif constraint == "smooth":
        v = list(np.ones(len(b_diff_diff), dtype=np.int)) #+ [0,0]
    elif constraint == "tps":
        v = list(np.ones(len(beta), dtype=np.int))
    elif constraint == "peak":
        assert (y is not None), "Include y in check_constraints for penalty=[peak]"
        assert (basis is not None), "Include basis in check_constraints for penalty=[peak]"

        peak, properties = find_peaks(x=y, distance=int(len(y)))
        border = np.argwhere(basis[peak,:] > 0)
        left_border_spline_idx = int(border[0][1])
        right_border_spline_idx = int(border[-1][1])
        v_inc = [0 if i > 0 else 1 for i in b_diff[:left_border_spline_idx]]
        v_dec = [0 if i < 0 else 1 for i in b_diff[right_border_spline_idx:]]
        v_plateau = np.zeros(right_border_spline_idx - left_border_spline_idx + 1)
        v = np.concatenate([v_inc, v_plateau, v_dec]) 
        
        # delete the last two entries
        v = v[:-2]
        
    elif constraint == "valley":
        assert (y is not None), "Include y in check_constraints for penalty=[valley]"
        assert (basis is not None), "Include basis in check_constraints for penalty=[peak]"

        peak, properties = find_peaks(x= -1*y, distance=int(len(y)))
        border = np.argwhere(basis[peak,:] > 0)
        left_border_spline_idx = int(border[0][1])
        right_border_spline_idx = int(border[-1][1])
        v_dec = [0 if i < 0 else 1 for i in b_diff[:left_border_spline_idx:]]
        v_inc = [0 if i > 0 else 1 for i in b_diff[right_border_spline_idx:]]
        v_plateau = np.zeros(right_border_spline_idx - left_border_spline_idx + 1)
        v = np.concatenate([v_dec, v_plateau, v_inc])
        
        # delete the last two entries
        v = v[:-2]
    
    else:
        print(f"Constraint [{constraint}] not implemented!")
        return    
    
    V = np.diag(v)
    
    if print_idx:
        print("Constraint violated at the following indices: ")
        print([idx for idx, n in enumerate(v) if n == 1])
    return V

def check_constraint_full_model(model):
    """Checks if the coefficients in the model violate the given constraints.
    
    Parameters:
    -------------
    model : class Model() 
    
    Returns:
    -------------
    v : list   - list of boolean wheter the constraint is violated. 
    """

    assert (model.coef_ is not None), "Please run Model.fit(X, y) first!"
    v = []

    for i, smooth in enumerate(model.smooths):
        beta = model.coef_[model.coef_list[i]:model.coef_list[i+1]]
        constraint = smooth.constraint
        V = check_constraint(beta, constraint=constraint, y=model.y, basis=model.basis)
        v += list(np.diag(V))
    
    return np.array(v, dtype=np.int)    
    
def bar_chart_of_coefficient_difference_dataframe(df):
    """Takes the dataframe Model.df_beta and plots a bar chart of the rows. """

    fig = go.Figure()
    xx = df.columns[1:]
    
    for i in range(df.shape[0]):
        fig.add_trace(go.Bar(x=xx, y=np.diff(df.iloc[i]), name=f"Iteration {i}"))
        
    fig.update_xaxes(
        showgrid=True,
        ticks="outside",
        tickson="boundaries",
        ticklen=20
    )
    fig.update_layout(title="Difference in coefficients", )
    fig.show()        
        
def line_chart_of_coefficient_dataframe(df):
    """Takes the dataframe Model.df_beta and plots a line chart of the rows. """

    fig = go.Figure()
    x = np.arange(df.shape[1])

    for i in range(df.shape[0]):
        fig.add_trace(go.Scatter(x=x, y=df.iloc[i], name=f"Iteration {i}",
                                mode="lines"))

    fig.update_layout(title="Coefficients at different iterations",)
    fig.show()

