#!/usr/bin/env python
# coding: utf-8


import plotly.graph_objects as go
import numpy as np
import pandas as pd
import pprint
import copy
from numpy.linalg import lstsq
from scipy.linalg import block_diag
from scipy.signal import find_peaks
from scipy.stats import t
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y 
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import ParameterGrid

from .smooth import Smooths as s
from .smooth import TensorProductSmooths as tps
from .penalty_matrix import PenaltyMatrix
from .utils import check_constraint, check_constraint_full_model
from .utils import test_model_against_constraint

class StarModel(BaseEstimator):
    """Implementation of a structured additive regression model.

    Fit data in the form of (X, y) using the structured additive regression model of
    Fahrmeir, Regression 2013 Cha. 9. Also incorporates prior knowledge in form of 
    shape constraints, e.g. increasing, decreasing, convex, concave, peak, valley. 

    """

    def __init__(self, description):
        """Initialitation of the Model using the nested tuple 'description'. 

        Parameters
        ----------
        description: tuple
            Description of the model

            Note
            ----       
            Using the scheme for the description tuple:

            (type of smooth, type of constraint, number of knots, lambdas, knot_type), 
            
            e.g. description = 
            ( 
            
                ("s(1)", "smooth", 25, (1, 100), "equidistant"),

                ("s(2)", "inc", 25, (1, 100), "quantile"), 
                
                ("t(1,2)", "smooth", [5,5], (1, 100), "quantile"), 
            
            ).
                        
        """

        self.description_str = description
        self.description_dict = {
            t: {"constraint": p, "n_param": n, 
                "lam" : {"smoothness": l[0], "constraint": l[1]},
                "knot_type": k
               } 
            for t, p, n, l, k  in self.description_str}
        self.smooths_list = list(self.description_dict.keys())
        self.smooths = None
        self.coef_ = None
        
    def __str__(self):
        """Pretty printing of the model description."""

        pp = pprint.PrettyPrinter()
        return pp.pformat(self.description_dict)

    
    def create_basis(self, X, y=None):
        """Create the BSpline basis for the data X.
        
        Reads through the description dictionary and creates either
        Smooths or TensorProductSmooths with given parameters according 
        to the description dictionary using the data given in X.

        Parameters
        ----------
        X : np.ndarray
            Data of size (n_samples, n_dimensions).
        y:  array 
            Target data of size (n_samples, ) for peak/valley penalty. 
        
        """

        self.smooths = list()
        for k,v in self.description_dict.items():
            if k.startswith("s"):    
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
            elif k.startswith("t"):
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
        self.smoothness_penalty_list = [
            s.lam["smoothness"] * s.smoothness_matrix(n_param=s.n_param).T @ s.smoothness_matrix(n_param=s.n_param) for s in self.smooths]
        #  self.smoothness_penalty_matrix is already lambda * S.T @ S
        self.smoothness_penalty_matrix = block_diag(*self.smoothness_penalty_list)

        n_coef_list = [0] + [np.product(smooth.n_param) for smooth in self.smooths]
        self.coef_list = np.cumsum(n_coef_list)            
                                                
    def create_constraint_penalty_matrix(self, beta_test=None):
        """Create the penalty block matrix specified in self.description_str.
        
        

        Parameters
        ----------
        beta_test : array
            Array of coefficients to be tested against the given constraints.
                
        """
        #  Looks like: ----------------------------------- 
        #             |lam1*p1 0        0        0      |  
        #             |0       lam2*p2  0        0      | = P 
        #             |0       0        lam3*p3  0      |
        #             |0       0        0        lam4*p4|
        #             -----------------------------------
        #  where P is a a matrix according to the specified penalty. The matrix P.T @ V @ P is then used, where
        #  V is a diagonal matrix of 0s and 1s, where a 1 is placed if the constraint is violated. 
        #  TODO:
        #     [x]  include the weights !!! 
        #     [x]  include TPS smoothnes penalty
        #     [ ]  include TPS shape penalty
        
        assert (self.smooths is not None), "Run Model.create_basis() first!"
        assert (beta_test is not None), "Include beta_test!"
        
        self.constraint_penalty_list = []        
        for idx, smooth in enumerate(self.smooths):
            b = beta_test[self.coef_list[idx]:self.coef_list[idx+1]]
            P = smooth.penalty_matrix
            V = check_constraint(beta=b, constraint=smooth.constraint)
            self.constraint_penalty_list.append(smooth.lam["constraint"] * P.T @ V @ P )
        #  self.constraint_penalty_matrix is already lambda P.T @ V @ P.T
        self.constraint_penalty_matrix = block_diag(*self.constraint_penalty_list)
    
    def calc_LS_fit(self, X, y):
        """Calculate the basis least squares fit without penalties.

        Uses np.linalg.lstsq to calculate the least squares fit for the basis given
        after create_basis() is run. 

        Parameters
        ----------
        X : np.ndarray
            Data of size (n_samples, n_dimensions)
        y : array
            Target data of size (n_samples,).
        
        Returns:
        -------------
        self  : object
            The trained model.
        
        """
        
        self.create_basis(X=X, y=y.ravel())    
        fitting = lstsq(a=self.basis, b=y, rcond=None)
        beta_0 = fitting[0].ravel()
        self.coef_, self.LS_coef_ = beta_0, beta_0        
        return self

    def create_df_for_beta(self, beta_init=None):
        """Create a DataFrame to save the calculated coefficients during the iteration.

        DataFrame contains one colume for each coefficient and one row for
        each iteration. 

        Parameters
        ----------
        beta_init : array
            Coefficient array to initilize the dataframe. 

        Returns
        -------
        df : pd.DataFrame
            Coefficient DataFrame.
        
        """

        col_name = [ f"b_{i}" for i in range(len(beta_init))]        
        df = pd.DataFrame(columns=col_name)
        d = dict(zip(col_name, beta_init))
        df = df.append(pd.Series(d), ignore_index=True)
        return df

    def fit(self, X, y, plot_=True, max_iter=5):
        """Calculate the PIRLS fit for data (X, y).

        Calculate the penalized iterative reweighted least squares (PIRLS) fit for
        the data (X, y). For further information, see Hofner B., 2012.
        
        Parameters
        ----------
        X : np.ndarray
            Data of size (n_samples, n_dimensions).
        y : array
            Target data of size (n_samples, ).
        max_iter : int
            Maximal number of iterations of PIRLS.
        plot_ : boolean
            Indicatior whether to plot the results.
        
        Returns 
        -------
        self : object
            Returns the fitted model.

        """
        #  TODO:
        #     [x] check constraint violation in the iterative fit
        #     [x] incorporate TPS in the iterative fit

        if len(X.shape) == 1:
            X = X.values.reshape((-1,1))
        X, y = check_X_y(X, y.ravel())
        self = self.calc_LS_fit(X=X, y=y)
        df = self.create_df_for_beta(beta_init=self.coef_)
        
        for _ in range(max_iter):
            self.create_constraint_penalty_matrix(beta_test=self.coef_)
            DVD = self.constraint_penalty_matrix
            DD = self.smoothness_penalty_matrix
            BB, By = self.basis.T @ self.basis, self.basis.T @ y
            v_old = check_constraint_full_model(model=self)
            beta_new = (np.linalg.pinv(BB + DD + DVD) @ By).ravel()
            self.coef_ = beta_new                       
            v_new = check_constraint_full_model(model=self)
            df = df.append(pd.DataFrame(data=beta_new.reshape(1,-1), columns=df.columns))
            delta_v = np.sum(v_new - v_old)
            #  change the criteria to the following: 
            #  print("Differences at the following coefficients: ", np.argwhere(v_old != v_new))
            if delta_v == 0: 
                break

        self.df = df
        self.mse = np.round(mean_squared_error(y, self.basis @ self.coef_), 7)       
        if plot_: 
            self.plot_fit(X=X, y=y).show()
            print(f"Violated Constraints: {np.sum(check_constraint_full_model(model=self))} from {len(self.coef_)} ")
        return self

    def weighted_fit(self, X, y, critical_point=None, plot_=True, max_iter=5):
        """Calculate a weighted PIRLS fit that goes through the critical point. 

        Parameters
        ----------
        X : np.ndarray
            Data of size (n_samples, n_dimensions).
        y : array
            Target data of size (n_samples, ).
        max_iter : int
            Maximal number of iterations of PIRLS.
        plot_ : boolean
            Indicatior whether to plot the results.
        critical_point: array
            Critical points for the weighted fit. More than one are allowed.

        Returns 
        -------
        self : object
            Returns the fitted model.        

        """
        assert (X.shape[1] == 1), "Only 1-d fits currently possible!"
        if critical_point is None:
            print("Insert a critical point!")
            return None
        x = X
        w = np.ones(x.shape[0]+len(critical_point))
        for cp in critical_point:
            x = np.append(x, cp[0])
            x.sort()
            idx_x = np.where(x == cp[0])[0][0]
            y = np.insert(arr=y, obj=idx_x, values=cp[1])

            w[idx_x] = 1000
            print("w[idx_x_new] = ", w[idx_x])
    
        x, y = x.reshape(-1,1), y.ravel()
        w = np.diag(w)

        X, y = check_X_y(x, y.ravel())
        self = self.calc_LS_fit(X=X, y=y)
        df = self.create_df_for_beta(beta_init=self.coef_)
        
        for _ in range(max_iter):
            self.create_constraint_penalty_matrix(beta_test=self.coef_)
            DVD = self.constraint_penalty_matrix
            DD = self.smoothness_penalty_matrix
            BwB, Bwy = self.basis.T @ w @ self.basis, self.basis.T @ (np.diag(w) * y)
            v_old = check_constraint_full_model(model=self)
            beta_new = (np.linalg.pinv(BwB + DD + DVD) @ Bwy).ravel()
            v_new = check_constraint_full_model(model=self)
            self.coef_ = beta_new                       
            df = df.append(pd.DataFrame(data=beta_new.reshape(1,-1), columns=df.columns))
            delta_v = np.sum(v_new - v_old)
            if delta_v == 0: 
                break

        self.df = df
        self.mse = mean_squared_error(y, self.basis @ self.coef_)       
        if plot_: 
            self.plot_fit(X=X, y=y).show()
            print(f"Violated Constraints: {np.sum(check_constraint_full_model(model=self))} from {len(self.coef_)} ")
        return self      
    
    def calc_cov_beta(self):
        """Calculate the covariance matrix for the coefficients.

        Returns
        -------
        cov_beta : ndarray
            REML estimator for the covariance matrix for the coefficients.

        """
        
        check_is_fitted(self, attributes="coef_", msg="Estimator is not fitted when using calc_cov_beta()!")
        XtX_inv = np.linalg.pinv(self.basis.T @ self.basis)
        n, p = self.basis.shape[0], self.basis.shape[1]
        cov_beta = (1 / (n-p)) * self.mse * XtX_inv
        return cov_beta


    def calc_confidence_intervals(self, alpha=0.05):
        """Calculates the lower and upper confidence interval for the fitted coefficients coef_.
        
        Currently, only 1D is possible.

        Parameters
        ----------
        alpha : float
            Confidence interval level.
        
        Returns
        -------
        beta_lower : array
            Lower confidence bound for the coefficients.
        beta_upper : array
            Upper confidence bound for the coefficients.

        """

        check_is_fitted(self, attributes="coef_", msg="Estimator is not fitted when using calc_confidence_intervals()!")
        n, p = self.basis.shape[0], self.basis.shape[1]
        t_value = t.ppf(q=1-alpha/2, df=n-p)
        se_beta = np.sqrt(np.diag(self.calc_cov_beta()))

        beta_lower = self.coef_ - t_value * se_beta
        beta_upper = self.coef_ + t_value * se_beta
        return beta_lower, beta_upper

    def calc_single_point_prediction_interval(self, x_pred, alpha=0.05):
        """Calculates the prediction interval for a single point according to Fahrmeir, Chap. 3.3.2.
        
        Currently, only 1D is possible. 

        Parameters
        ----------
        x_pred : float
            Point to calculate the prediction interval for.
        alpha : float
            Prediction interval level.
        Returns
        -------
        lower_pi, upper_pi : tuple
            Lower and upper prediction interval bounds. 

        """

        check_is_fitted(self, attributes="coef_", msg="Estimator is not fitted when using calc_single_point_prediction_interval()!")
        knots = self.smooths[0].knots
        n, p = self.basis.shape[0], self.basis.shape[1]
        t_value = t.ppf(1-alpha/2, df=n-p)
        XtX_inv = np.linalg.inv(self.basis.T @ self.basis)
        xi = []
        for i in range(self.basis.shape[1]):
            xi.append(self.smooths[0].bspline(x=x_pred, knots=knots, i=i, m=2))
        xi = np.array(xi)
        pred = xi @ self.coef_
        pred_interval = t_value * np.sqrt( self.mse / (n-p)) * np.sqrt(1 + xi.T @ XtX_inv @ xi)
        return (pred-pred_interval, pred+pred_interval)

    def calc_prediction_interval(self, x_pred, alpha=0.05, fig=False):
        """Calculate and plot the prediction interval for multiple points according to Fahrmeir, Chap. 3.3.2. 
        
        Currently, only 1D is possible.
        
        Parameters
        ----------
        x_pred : array
            Array to calculate the prediction interval for.
        alpha : float
            Prediction interval level
        fig : go.Figure()
            Figure to plot the prediction interval in.
        
        Returns
        -------
        y_lower, y_upper : tuple
            Lower and upper prediction interval bounds.

        """
        y_lower, y_upper = [], []
        for x in x_pred:
            pi = self.calc_single_point_prediction_interval(x_pred=x)
            y_lower.append(pi[0])
            y_upper.append(pi[1]) 
        if fig:
            fig.add_trace(go.Scatter(x=x_pred, y=y_lower, name="Lower PI", mode="lines", line=dict(dash="dashdot", color="violet")))
            fig.add_trace(go.Scatter(x=x_pred, y=y_upper, name="Upper PI", mode="lines", line=dict(dash="dashdot", color="violet")))
            return fig
        return (y_lower, y_upper)

    def plot_confidence_intervals(self, alpha=0.05, fig=None, y=None, x=None):
        """Plot the confidence intervals into fig. """

        check_is_fitted(self, attributes="coef_", msg="Estimator is not fitted when using plot_confidence_intervals()!")
        beta_lower, beta_upper = self.calc_confidence_intervals(alpha=0.05, y=y)
        fig.add_trace(go.Scatter(
            x=x.ravel(), y=self.basis @ beta_lower, name="Lower Confidence Interval Bound", mode="lines",
            line=dict(dash="dash", color="green")))
        fig.add_trace(go.Scatter(
            x=x.ravel(), y=self.basis @ beta_upper, name="Upper Confidence Interval Bound", mode="lines",
            line=dict(dash="dash", color="green")))
        return fig

    def plot_cost_function_partition(self, y, print_=True):
        """Plot the partition of the cost function.

        Q(beta) = || y - X*beta ||^2 + lam_1*J_smooth(betea) + lam_2*J_constraint(beta). 
        Currently only in 1D.

        Parameters:
        -----------
        y : np.array
            Target data to calculate the MSE.
        print_ : bool
            Indicator whether to print more information.

        Returns:
        --------
        fig : plotly.graph_objs.Figure

        """

        check_is_fitted(self, attributes="coef_", msg="Estimator is not fitted when using plot_cost_function_partition()!")
        #  smoothness part of the cost function
        S = PenaltyMatrix().smoothness_matrix(n_param=self.smooths[0].n_param)
        #  lam_s = self.description_dict["s(1)"]["lam"]["smoothness"]
        #  lam_c = self.description_dict["s(1)"]["lam"]["constraint"]
        J = self.coef_.T @ S.T @ S @ self.coef_
        P = self.smooths[0].penalty_matrix
        J_constr = self.coef_.T @ P.T @ np.diag(check_constraint_full_model(model=self)) @ P @ self.coef_
        D = mean_squared_error(y, self.basis @ self.coef_)

        J_pre = self.LS_coef_ @ S.T @ S @ self.LS_coef_
        J_constr_pre = self.LS_coef_ @ P.T @ P @ self.LS_coef_
        D_pre = mean_squared_error(y, self.basis @ self.LS_coef_)

        fig = go.Figure(data=[go.Pie(labels=["Data", "Smoothness", "Constraint"], values=[D, J, J_constr])])
        fig.update_layout(title="Cost function partition of constraint LS fit")

        fig2 = go.Figure(data=[go.Pie(labels=["Data", "Smoothness", "Constraint"], values=[D_pre, J_pre, J_constr_pre])])
        fig2.update_layout(title="Cost function partition of pure LS fit")

        if print_:
            print("Values without lambdas: ")
            print("J(beta) = ", J)
            print("J_constr(beta) = ", J_constr)
            print("D(beta) = ", D)
        
            print("Values without lambdas and LS fit: ")
            print("J_pre(beta) = ", J_pre)
            print("J_constr_pre(beta) = ", J_constr_pre)
            print("D_pre(beta) = ", D_pre)

        return fig, fig2

    def plot_fit(self, X, y):
        """Plot the fitted model and the given data.

        Only possible for 1d and 2d data X. 
        
        Parameters
        ----------
        X : nd.array
            Data of size (n_samples, n_dimensions).
        y : array
            Target data of size (n_samples, ).

        Returns
        -------
        fig : plotly.graph_objs.Figure 

        """

        check_is_fitted(self, attributes="coef_", msg="Estimator is not fitted when using plot_fit()!")
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
        fig.update_layout(autosize=True) 
        return fig

    def plot_fit_and_LS_fit(self, X, y):
        """Plot the fitted values model and the least squares fit without constraint.

        Parameters
        ----------
        X : nd.array
            Data of size (n_samples, n_dimensions)
        y : array
            Target data of size (n_samples, )

        Returns
        -------
        fig : plotly.graph_objs.Figure 

        """

        X, y = check_X_y(X=X, y=y)
        fig = self.plot_fit(X=X, y=y)
        fig.add_trace(go.Scatter(
            x=X, y=self.basis @ self.LS_coef_, mode="markers+lines", name="Least Squares Fit")
        )
        return fig       

    def predict(self, X_pred, extrapol_type="zero", depth=3):
        """Prediction of the trained model on the data in X.
        
        Currently only 1-DIMENSIONAL prediction possible !!!

        Parameters
        ----------
        X_pred : np.ndarray
            Data of shape (n_samples,) to predict values for.
        extrapol_type: str
            Indiactor of the extrapolation type. 
        depth : int
            Indicates how many coefficients are used for the linear extrapolation.

        Returns
        -------
        pred : array
            Returns the predicted values. 
 
        """

        check_is_fitted(self, attributes="coef_", msg="Estimator is not fitted when using predict()!")
        y_pred = []
        for x in X_pred:
            if 0 <= x <= 1:
                y_pred.append(self.smooths[0].spp(
                    sp=x, coef_=self.coef_, knots=self.smooths[0].knots))
            else:
                y_pred.append(self.extrapolate(x_exp=x, type_=extrapol_type, depth=3))
        y_pred = np.array([y_pred])
        return y_pred.ravel()

    def predict_single_point(self, x_sp):
        """Fast single point prediction.

        Currently only 1-DIMENSIONAL prediction possible !!!

        Parameter
        ----------
        x_sp : float
            Single point data.

        Returns
        -------
        y_sp : float
            Predicted value.

        """

        return self.smooths[0].spp(
            sp=x_sp, coef_=self.coef_, knots=self.smooths[0].knots)


    def extrapolate(self, x_exp, type_="constant", depth=5):
        """ Evaluate the extrapolation value for the given x_exp. 

        Parameters
        ----------
        x_exp : array
            Data to calculate the extrapolation for.
        type_ : str
            Describes the extrapolation type, either "constant", "linear", "zero".
        depth : int
            Describes ow many coefficients are taken into account for the linear extrapolation.

        Returns
        -------
        y_extrapolate : array
            Extrapolation value.

        """

        check_is_fitted(self, attributes="coef_", msg="Estimator is not fitted when using extrapolate()!")
        assert (type_ in ["constant", "linear", "zero"]), f"Typ_ '{type_}' not supported!"
        direction = "left" if x_exp < 0 else "right"
        if direction == "left" and type_ in ["constant", "linear"]:
            #print("left + const/lin")
            y_boundary = self.predict_single_point(x_sp=0)
            k = np.mean(self.coef_[:depth])
        elif direction == "right" and type_ in ["constant", "linear"]:
            #print("right + const/lin")
            y_boundary = self.predict_single_point(x_sp=1)
            k = np.mean(self.coef_[-depth:])

        if type_ == "constant":
            #print("const")
            y_extrapolate = y_boundary
        elif type_ == "linear":
            #print("linear")
            dx = x_exp-1 if x_exp > 0 else np.abs(x_exp)
            y_extrapolate = y_boundary + dx * k
        elif type_ == "zero" and direction == "left":
            #print("left + zero")
            y_extrapolate = self.smooths[0].left_exterior_spp(sp=x_exp, coef_=self.coef_, width=depth)
        elif type_ == "zero" and direction == "right":
            #print("right + zero")
            y_extrapolate = self.smooths[0].right_exterior_spp(sp=x_exp, coef_=self.coef_, width=depth)
        else:
            return

        return y_extrapolate

    def calc_hat_matrix(self):
        """Calculates the hat matrix (influence matrix) of the fitted model.
        
        The matrix is given by S = Z(Z'Z + S'S + P'P) Z', where Z is the Bspline 
        basis (n_samples x n_param), S'S is the smoothing constraint penalty matrix (n_param x n_param) 
        and P'P is the user-defined constraint penalty matrix. 

        Returns
        -------
        H : np.ndarray
            Hat matrix of size (n_data x n_data).

        """

        if hasattr(self, "basis") and hasattr(self, "constraint_penalty_matrix") and hasattr(self, "smoothness_penalty_matrix"):
            Z = self.basis
            P = self.constraint_penalty_matrix
            S = self.smoothness_penalty_matrix
            H = Z @ np.linalg.pinv(Z.T @ Z + + S + P) @ Z.T
        else:
            print("Fit the model to data!")
        return H

    def generate_GCV_parameter_list(self, n_grid=5, p_min=1e-4):
        """Generates the exhaustive parameter list for the GCV. 

        Parameters
        ----------
        n_grid : int
            Number of distinct parameter values to try. 
        p_min : float
            Minimum parameter value.

        Returns
        -------
        grid : ParameterGrid()
            Returns an iterator.
        
        """

        params = dict()
        for k in list(self.description_dict):
            for k2, v2 in self.description_dict[k]["lam"].items():
                params[k+"_"+k2] = v2        

        for k in params.keys():
            spacing = np.geomspace(p_min, p_min*10**n_grid, n_grid, endpoint=False)
            lam_type = k[k.find("_")+1]
            if lam_type == "s":
                params[k] = spacing
            elif lam_type == "c":
                params[k] = spacing * 1e6

        grid = ParameterGrid(params)

        return grid

    def calc_GCV_score(self, y):
        """Calculates the generalized cross validation score according to Fahrmeir, Regression 2013 p.480.
        
        Parameter
        ---------
        y : array
            Target data of size (n_samples,). 

        Returns
        -------
        GCV : float
            Generalized cross validation score.
        
        """

        y_hat = self.basis @ self.coef_
        n = y.shape[0]
        trace = np.trace(self.calc_hat_matrix())
        GCV = (1/n) * np.sum((y.ravel() - y_hat.ravel())**2 / (1 - trace/n))
        return GCV

    def calc_GCV(self, X, y, n_grid=5, p_min=1e-4, plot_=False):
        """Carry out the generalized cross validation for the model.

        This iterates over a grid with n_grid**(n_constraint*2). For each smooth, there are
        2 lambdas given (one for smoothing, one for the constraint), so the size of the grid 
        increases EXPONENTIALLY with this number. 

        Parameters
        ----------
        X : nd.array
            Data of size (n_samples, n_dimensions).
        y : array
            Target data of size (n_samples,).
        n_grid : int
            Number of distinct parameter values to try.
        p_min : float
            Minimum parameter value.
        plot_ : bool
            Indicator of whether to plot the fit.
        
        Returns
        -------
        self : object
            Returns the model with the best set of lambdas according to the GCV score.

        """


        X, y = check_X_y(X=X, y=y)
        grid = self.generate_GCV_parameter_list(n_grid=n_grid, p_min=p_min)
        gcv_scores, violated_constraints_list = [], []
        for params in grid:
            for k,v in params.items():
                for k2 in self.description_dict.keys():
                    if k[:k.find("_")] == k2:
                        self.description_dict[k2]["lam"][k[k.find("_")+1:]] = v
            self.fit(X=X, y=y.ravel(), plot_=plot_)
            if plot_:
                print(f"Parameters: {params}")

            ccfm = check_constraint_full_model(model=self)
            gcv_new = self.calc_GCV_score(y=y)          
            #  add a penalty for violating constraints like gcv = gcv(1+penalty), where
            #  penalty < 1
            gcv_new = gcv_new * (1 + np.sum(ccfm) / len(self.coef_))
            gcv_scores.append(gcv_new)
            violated_constraints_list.append(ccfm)

        gcv_best = list(grid)[np.argmin(gcv_scores)] 
        print(f"Best fit parameter according to adapted-GCV score: {gcv_best}")
        print(f"Violated Constraints: {np.sum(violated_constraints_list[np.argmin(gcv_scores)])} from {len(self.coef_)}")
        self.set_params_after_gcv(params=gcv_best)
        self.fit(X=X, y=y, plot_=plot_)
        self.gcv_best = gcv_best
        return self
 
    def get_params(self, deep=True):
        """Returns a dict of __init__ parameters. 
        
        If deep==True, also return parameters of sub-estimators (can be ignored).

        """

        return self.description_dict

    def set_params(self, params):
        """Sets the parameters of the object to the given in params.

        Use self.get_params() as template. Only run after .create_basis(X=X) is run.

        Parameters
        ----------
        params : nested dict
            Should have the form of self.get_params
        
        """

        for smooth, param  in zip(self.smooths, params.items()):
            for key, value in param[1].items():
                if hasattr(smooth, key):
                    setattr(smooth, str(key), value)

    def set_params_after_gcv(self, params):
        """Sets the lambda parameters after the GCV search. 

        Only run inside of calc_GCV.

        Parameters
        ----------
        params  : dict
            Best parameter combination found by GCV.
        
        """
        
        descr_dict = copy.deepcopy(self.description_dict)
        for k in params.keys():
            idx = k.find("_")
            s = k[:idx]
            t = k[idx+1:]
            descr_dict[s]["lam"][t] = params[k]
        self.description_dict = descr_dict



    def eval_metric(self, X_test=None, y_test=None, precision=5):
        """Evaulate the metric M = MSE_prediction + ICP. 
        
        ICP = Invalid Constraint percentage. 
        
        Parameters
        ----------
        X_test : array
            Data to calculate the prediction MSE for.
        y_test : arrray
            Data to calculate the prediciton MSE for.
        precision: int
            Precision of the metric

        Returns
        -------
        metric : float
            Value of the metric.

        """

        test = test_model_against_constraint(model=self, plot_=False)
        ICP = test.sum() / len(test)
        
        y_pred = self.predict(X_pred=X_test)
        mse_test = mean_squared_error(y_pred, y_test)

        metric = 1*mse_test + 1*ICP
        return np.round(metric, precision)