#!/usr/bin/env python
# coding: utf-8



import pprint
import copy
import sys
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.linalg import lstsq
from scipy.linalg import block_diag
from scipy.stats import norm, probplot
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y 
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import ParameterGrid
from plotly.subplots import make_subplots
from statsmodels.nonparametric.smoothers_lowess import lowess

from .smooth import Smooths as s
from .smooth import TensorProductSmooths as tps
from .penalty_matrix import PenaltyMatrix
from .utils import check_constraint, check_constraint_full_model
from .utils import test_model_against_constraints

class StarModel(BaseEstimator):
    """Implementation of a structured additive regression model.

    Fit data in the form of (X, y) using the structured additive regression model of
    Fahrmeir, Regression 2013 Cha. 9. Also incorporates prior knowledge in form of 
    shape constraints, e.g. increasing, decreasing, convex, concave, peak, valley. 

    """

    def __init__(self, description=None):
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
        if description == None:
            print("Please include a model description of the form:")
            print("Smooth of Variable i -- Constraint Type -- Number of Splines -- (lam_smoothnes, lam_constraint) -- Grid Placement)")
            sys.exit()

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
        smooths = dict()
        for k,v in self.description_dict.items():
            if k.startswith("s"):
                smooths[k] = s(x_data=X[:,int(k[2])-1],  n_param=v["n_param"], constraint=v["constraint"], 
                                y=y,lambdas=v["lam"], type_=v["knot_type"])
            elif k.startswith("t"):
                smooths[k] = tps(x_data=X[:, [int(k[2])-1, int(k[4])-1]],  n_param=list(v["n_param"]), 
                                constraint=v["constraint"], y=y, lambdas=v["lam"], type_=v["knot_type"])
        self.basis = np.concatenate([v.basis for v in smooths.values()], axis=1)
        # add a normalization factor if different numbers of splines are used for different smooths
        smoothnes_penalty_list = [(1/v.smoothness.shape[1]) * v.lam["smoothness"] * v.smoothness for v in smooths.values()]
        self.smoothness_penalty_matrix = block_diag(*smoothnes_penalty_list)
        n_coef_list = [0] + [np.product(smooth.n_param) for smooth in smooths.values()]
        self.coef_list = np.cumsum(n_coef_list)
        self.smooths = smooths          
                                                
    def create_constraint_penalty_matrix(self):
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
               
        cp_list = []
        self.constraint_penalty_matrix = None
        for v in self.smooths.values():
            b = v.coef_
            P = v.penalty_matrix
            V = check_constraint(beta=b, constraint=v.constraint, smooth_type=type(v))
            cp_list.append((1/P.shape[1])*v.lam["constraint"] * (P.real.T @ V @ P.real).round(4))
        self.constraint_penalty_matrix = block_diag(*cp_list)

    
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
        try:    
            fitting = lstsq(a=self.basis, b=y, rcond=None)
        except np.linalg.LinAlgError:
            beta_0 = np.linalg.pinv(self.basis.T @ self.basis) @ self.basis.T * y

        beta_0 = fitting[0].ravel()
        self.coef_, self.LS_coef_ = beta_0, beta_0        
        #  update the coefficient values for each smooth
        for i, v in enumerate(self.smooths.values()):
            v.coef_ = self.coef_[self.coef_list[i]:self.coef_list[i+1]]
        return self

    def fit(self, X, y, plot_=True, max_iter=10, cp=None, weight=1000, verbose=False, global_pos=False):
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
        cp : None or array
            Points for weighted fit, if None they are ignored.
        weight : int
            Weight value for weighted fit. 
        verbose: boolean
            Verbose printing.
        global_pos : boolean
            Whether the global positivity constraint should be applied or not.
        
        Returns 
        -------
        self : object
            Returns the fitted model.

        """
        if len(X.shape) == 1:
            X = X.values.reshape((-1,1))
        if cp is not None:
            X, y, w = self.add_weighted_points(X=X, y=y, Xw=cp[:,:-1], yw=cp[:,-1])
        else:
            w = np.ones(X.shape[0])
        X, y = check_X_y(X, y.ravel())
        self = self.calc_LS_fit(X=X, y=y)
        df = self.create_df_for_beta(beta_init=self.coef_)
        
        for iter in range(max_iter):
            self.create_constraint_penalty_matrix()
            DVD = self.constraint_penalty_matrix
            DD = self.smoothness_penalty_matrix
            BwB, Bwy = self.basis.T @ np.diag(w) @ self.basis, self.basis.T @ (w * y)

            if global_pos:
                ypred = self.basis @ self.coef_
                V = np.diag(ypred < 0).astype(int)
                # get maximum lambda and multiply it by 10
                lam = 10*max(list(map(lambda x: max(x.values()),  [v["lam"] for v in self.description_dict.values()])))
                BVB = lam * self.basis.T @ V @ self.basis
                print(f"Nr of predictions smaller than zero before iteration {iter}: {np.sum(np.diag(V))}")
            else:
                BVB = np.zeros((self.basis.shape[1], self.basis.shape[1]))
            
            v_old = check_constraint_full_model(model=self)
            beta_new = (np.linalg.pinv(BwB + DD + DVD + BVB) @ Bwy).ravel()
            self.coef_ = beta_new       
            #  update the coefficient values for each smooth
            for i, v in enumerate(self.smooths.values()):
                v.coef_ = self.coef_[self.coef_list[i]:self.coef_list[i+1]]
            v_new = check_constraint_full_model(model=self)
            df.loc[i+1] = self.coef_
            delta_v = np.sum(v_new - v_old)

            if verbose:
                print(f"Iteration {iter+1}".center(20,"-"))
                print(f"v_old = {sum(v_old)}".center(20, "-"))
                print(f"v_new = {sum(v_new)}".center(20, "-"), "\n")
            elif verbose and (delta_v == 0):
                print("PIRLS converged!".center(20, "-"))
                break
            else:
                if delta_v == 0: 
                    break
                else:
                    continue

        self.df = df
        self.mse = np.round(mean_squared_error(y, self.basis @ self.coef_), 7)       
        if plot_: 
            self.plot_fit(X=X, y=y).show()
            print(f"Violated Constraints: {np.sum(check_constraint_full_model(model=self))} from {len(self.coef_)} ")
        return self

    def predict(self, X, extrapol_type="zero", depth=10):
        """Prediction of the trained model on the data in X.

        !!! Only for normalized data !!!
        
        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples,) to predict values for.
        extrapol_type: 
            Indiactor of the extrapolation type. 
        depth : int
            Indicates how many coefficients are used for the linear extrapolation.

        Returns
        -------
        pred : array
            Returns the predicted values. 
 
        """

        y_pred = []
        for x in X:
            if np.all(x >= 0) and np.all(x <= 1):
                pred = []
                for k, v in self.smooths.items():
                    if k.startswith("s"):
                        pred.append(v.spp(sp=x[int(k[2])-1], coef_=v.coef_))
                    elif k.startswith("t"):
                        pred.append(v.spp(sp=x[int(k[2])-1:int(k[4])], coef_=v.coef_))
            else:
                pred.append(self.extrapolate(X=x, type_=extrapol_type, depth=depth))
            y_pred.append(sum(pred))
        y_pred = np.array([y_pred])
        return y_pred.ravel()

    def add_weighted_points(self, X, y, Xw, yw, weights=1000):
        """Add the points (Xw, yw) to the dataset and create a weight matrix.

        Parameter
        ---------
        X : np.ndarray
            Base data of shape (n_samples, n_dim).
        y : np.array
            Base target data of shape (n_samples, ).
        Xw : np.ndarray
            Weigthed data points of shape (n_weighted, dim).
        yw : np.array
            Weighted target data of shape (n_weighted, ).
        weights : int
            Weight value.
        
        Returns
        -------
        Xn : np.ndarray
            Adapted data set of shape (n_samples + n_weighted, dim).
        yn : np.array
            Adapted target data of shape (n_samples + n_weighted, ).
        W : np.ndarray
            Weight vector of shape (n_samples + n_weighted, ).

        """

        Xn = np.vstack((X, Xw))
        yn = np.append(y, yw)
        w = np.ones(Xn.shape[0])
        w[-1*Xw.shape[0]:] = weights
        return Xn, yn, w

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

    def predict_single_point(self, X):
        """Fast single point prediction.

        Parameter
        ----------
        X : array
            Single point data.

        Returns
        -------
        y_sp : float
            Predicted value.

        """
        y_pred = []
        for k, v in self.smooths.items():
            if k.startswith("s"):
                y_pred.append(v.spp(sp=X[int(k[2])-1], coef_=v.coef_))
            elif k.startswith("t"):
                y_pred.append(v.spp(sp=X[int(k[2])-1:int(k[4])], coef_=v.coef_))
        return sum(y_pred)

    def extrapolate(self, X, type_="constant", depth=10):
        """ Evaluate the extrapolation value for the given X. 

        Parameters
        ----------
        X : array
            Datapoint to calculate the extrapolation for, shape = (n_dim, )
        type_ : str
            Describes the extrapolation type, either "constant", "linear", "zero".
        depth : int
            Describes how many coefficients are taken into account for the linear extrapolation.

        Returns
        -------
        y_extrapolate : array
            Extrapolation value.

        """

        #  TODO: 
        #  [ ] - linear extrapolation in 2D
        #  [ ] - constant interpolation in 2D

        check_is_fitted(self, attributes="coef_", msg="Estimator is not fitted when using extrapolate()!")
        assert (type_ in ["constant", "linear", "zero"]), f"Typ_ '{type_}' not supported!"
        direction = "left" if np.any(X < 0) else "right"
        if direction == "left" and type_ in ["constant", "linear"]:
            #print("left + const/lin")
            y_boundary = self.predict_single_point(X=np.zeros(len(X)))
            k = np.mean(self.coef_[:depth])
        elif direction == "right" and type_ in ["constant", "linear"]:
            #print("right + const/lin")
            y_boundary = self.predict_single_point(X=np.ones(len(X)))
            k = np.mean(self.coef_[-depth:])

        if type_ == "constant":
            y_extrapolate = y_boundary
        elif type_ == "linear":
            dx = X-1 if X > 0 else np.abs(X)
            y_extrapolate = y_boundary + dx * k
        elif type_ == "zero" and direction == "left":
            y_extrapolate = self.predict_single_point(np.zeros(len(X))) * np.exp(-(X - 0)**2 / (1/depth))
        elif type_ == "zero" and direction == "right":
            y_extrapolate = self.predict_single_point(np.ones(len(X))) * np.exp(-(X - 1)**2 / (1/depth))
        else:
            return

        return y_extrapolate[0]

    def confidence_interval(self, X, alpha=0.05, bonferroni=True):
        """Calculate the confidence interval/band for nonparametric regression models.

        Based on Fahrmeir, Regression Chap. 8.1.8, p. 470. 
        Currently, only available for one dimension, even when the fit is multi-dimensional and
        only when the first dimenions is choose as smooth. 

        Parameters
        ----------
        X : array
            Point/Points to calculate the confidence interval/band for.
        alpha: float
            Confidence level.
        bonferroni: boolean
            Indicator whether to use the bonferroni correction for the confidence bands.

        Returns
        -------
        y_m : array
            Lower confidence interval/band.
        y_p : array
            Upper confidence interval/band.
        
        """

        y_p = self.predict(X=X)
        Z = self.basis
        lambda_k = self.constraint_penalty_matrix
        S = self.calc_hat_matrix()
        
        if bonferroni:
            m = norm.ppf(1 - alpha/(2*X.shape[0]))
        else:
            m = norm.ppf(1 - alpha/2)
        sigma_hat = np.sqrt(1 / (Z.shape[0] - np.trace(2*S - S @ S.T)) * Z.shape[0] * self.mse)

        z = np.empty((X.shape[0], len(self.coef_)))
        for coef_idx in range(len(self.smooths["s(1)"].coef_)):
            z[:,coef_idx] = self.smooths["s(1)"].bspline(x=X[:,0], knots=self.smooths["s(1)"].knots, i=coef_idx, m=2)
        
        s_t = z @ np.linalg.inv(Z.T @ Z + lambda_k) @ Z.T
        sqrt_sts = np.sqrt(np.diag(s_t @ s_t.T))
        
        y_m = y_p - m * sigma_hat * sqrt_sts
        y_p = y_p + m * sigma_hat * sqrt_sts
        
        return y_m, y_p


    def plot_confidence_intervals(self, alpha=0.05, fig=None, y=None, X=None, bonferroni=True):
        """Plot the confidence intervals into fig. """

        check_is_fitted(self, attributes="coef_", msg="Estimator is not fitted when using plot_confidence_intervals()!")
        y_m, y_p = self.confidence_interval(X=X, alpha=alpha, bonferroni=bonferroni)
        fig.add_trace(go.Scatter(
            x=X.ravel(), y=y_m, name="Lower Confidence Band", mode="lines",
            line=dict(dash="dash", color="green")))
        fig.add_trace(go.Scatter(
            x=X.ravel(), y=y_p, name="Upper Confidence Band", mode="lines",
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
        S = PenaltyMatrix().smoothness_matrix(n_param=self.smooths[0].n_param)
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
        GCV = (1/n) * np.sum( ((y.ravel() - y_hat.ravel()) / (1 - trace/n))**2   )
        return GCV

    def calc_CV_score(self, y):
        """Calculates the cross validation score according to Fahrmeir, Regression 2013 p.480.
        
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
        H = self.calc_hat_matrix()
        GCV = (1/n) * np.sum( ((y.ravel()-y_hat.ravel()) / (1-np.diag(H)) )**2 )
        return GCV


    def GCV_smoothingParameter(self, X, y, n_grid, p_min=1e-4, p_max=1e3):
        """Carry out the generalized cross validation for the smoothing paramter $\lambda_s$,
            as well as the cross validation.

        !!! Only for 1D.  !!!

        This iterates over a grid with n_grid**n_constraint. Returns the model for the best
        smoothing parameter and the constraint parameter set to 1000*best_smoothing_parameter.

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
        p_max : float
            Maximum parameter value.
        
        Returns
        -------
        self : object
            Returns the model with the best set of lambdas according to the GCV score.

        """
        X, y = check_X_y(X=X, y=y)
        # change lambda and calc model
        lam_grid = np.geomspace(p_min, p_max, num=n_grid).round(4)
        # CV for lambda-smoothness
        gcvs_smooth, cvs_smooth = {}, {}
        for l in lam_grid:
            self.description_dict["s(1)"]["lam"]["smoothness"] = l
            self.description_dict["s(1)"]["lam"]["constraint"] = 0
            self.fit(X=X, y=y, plot_=False)
            
            gcvs_smooth[str(l)] = self.calc_GCV_score(y=y)
            cvs_smooth[str(l)] = self.calc_CV_score(y=y)

        best_lam_smooth = float(min(gcvs_smooth, key=gcvs_smooth.get))
        self.description_dict["s(1)"]["lam"]["smoothness"] = best_lam_smooth
        self.description_dict["s(1)"]["lam"]["constraint"] = 1000*best_lam_smooth

        self.fit(X=X, y=y, plot_=False)
        self.gcv_smoothness = gcvs_smooth
        self.cv_smoothness = cvs_smooth

        return self

    def plot_GCV_curve(self):
        """Plot the GCV-score curve over the lambdas."""
        if hasattr(self, "gcv_smoothness"):
            lams = list(map(float, self.gcv_smoothness.keys()))
            gcv_scores = list(self.gcv_smoothness.values())
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=lams, y=gcv_scores, mode="lines", 
                    line=dict(width=2), name="Generalized Cross Validation Score"))
            fig.update_xaxes(type="log", title=r"$\lambda$")
            fig.show()
        else:
            print("Run Generalized Cross Validation First!")

    def plot_CV_curve(self):
        """Plot the CV-score curve over the lambdas."""
        if hasattr(self, "cv_smoothness"):
            lams = list(map(float, self.cv_smoothness.keys()))
            cv_scores = list(self.cv_smoothness.values())
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=lams, y=cv_scores, mode="lines", 
                    line=dict(width=2), name="Cross Validation Score"))
            fig.update_xaxes(type="log", title=r"$\lambda$")
            fig.show()
        else:
            print("Run Cross Validation First!")

    def calc_GCV(self, X, y, n_grid=5, p_min=1e-4, plot_=False):
        """Carry out the generalized cross validation for all $\lambda$ the model.

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
        for params in tqdm(grid):
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

    def MSPE(self, X=None, y=None, precision=5):
        """Calculate the mean squared prediction error.

        Parameter
        ---------
        X : np.ndarray
            Test data of shape (n_test, n_dim).
        y : np.array
            True target data of shape (n_test, ).
        precision : int
            Precision of the returned value.

        Returns
        -------
        mspe : float
            Mean squared prediction error.
        
        """
        ypred = self.predict(X=X)
        mspe = mean_squared_error(ypred, y)
        return mspe.round(decimals=precision)


    def eval_metric(self, X=None, y=None, precision=5):
        """Evaulate the metric M = MSE_prediction + ICP. 
        
        ICP = Invalid Constraint percentage. 
        
        Parameters
        ----------
        X : array
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

        ICP = test_model_against_constraints(model=self)
        
        y_pred = self.predict(X=X)
        mse_test = mean_squared_error(y_pred, y)

        metric = 1*mse_test + 1*ICP
        return np.round(metric, precision)

    def describe_data(self, X, y, col_names=False):
        """R like description of the dataset.

        Parameters
        ----------
        X : np.ndarray
            Data of size (n_samples, n_dim).
        y : array
            Target data of size (n_samples, ).
        col_names : list
            Optional, List of column names. 

        Returns
        -------
        df.describe : pd.DataFrame
            Returns the description of the dataset.

        """

        X = np.c_[X, y]
        df = pd.DataFrame(data={f"x{i}": X[:,i] for i in range(X.shape[1])})
        if col_names:
            df.columns = col_names
        else:
            df.rename(columns={f"x{X.shape[1]-1}":"y"})
        return df.describe()

    def plot_diagosticPlots(self, y=None, fname=False):
        fitted = self.basis @ self.coef_
        residuals = y.ravel() - fitted
        # hat matrix
        H = self.calc_hat_matrix()
        # estimated variance
        sigma_hat = np.sqrt(1 / (self.basis.shape[0] - np.trace(2*H - H @ H.T)) * self.basis.shape[0] * self.mse)
        # standardized residual
        residual_std = residuals / (sigma_hat * np.sqrt(1 - np.diag(H)))
        sqr_residual_std = np.sqrt(np.abs(residual_std))
        # quantile
        osm, _ = probplot(residual_std, dist="norm")
        # Cooks distance
        CD = residuals**2 / (np.trace(H) * self.mse) * (np.diag(H) / (1 - np.diag(H))**2)
        # high leverage points
        high_leverage_points = 2 *(np.trace(H) + 1) / len(fitted) < np.diag(H)
        # high influence points
        high_influence_points = 4 / (len(fitted) - np.trace(H) -1) < CD

        res_fit_1 = lowess(exog=fitted, endog=residuals, return_sorted=True)
        res_fit_2 = lowess(exog=fitted, endog=sqr_residual_std, return_sorted=True)
        res_fit_3 = lowess(exog=np.diag(H), endog=residual_std, frac=0.8, return_sorted=True)

        fig = make_subplots(rows=3, cols=2, 
                        subplot_titles=["Residual vs. Fitted", "Scale-Location", "Q-Q Normal", "Residual vs. Leverage", "Cook's Distance"],
                        column_widths=[0.5, 0.5], row_heights=[0.5, 0.5, 0.5], horizontal_spacing=0.15, vertical_spacing=0.15,
                            specs=[[{}, {}], [{}, {}], [{"colspan": 2}, None]])

        fig.add_trace(go.Scatter(x=fitted, y=residuals, mode="markers", marker=dict(color="royalblue")), row=1, col=1)
        fig.add_trace(go.Scatter(x=res_fit_1[:,0], y=res_fit_1[:,1], mode="lines", line=dict(dash="dash", color="red")), row=1, col=1)

        fig.add_trace(go.Scatter(x=fitted, y=sqr_residual_std, mode="markers", marker=dict(color="royalblue")), row=1,col=2)
        fig.add_trace(go.Scatter(x=res_fit_2[:,0], y=res_fit_2[:,1], mode="lines", line=dict(dash="dash", color="red")),row=1,col=2)

        fig.add_trace(go.Scatter(x=osm[0], y=osm[1], mode="markers", marker=dict(color="royalblue")), row=2, col=1)
        fig.add_trace(go.Scatter(x=[osm[0].min(), osm[0].max()], y=[osm[0].min(), osm[0].max()], mode="lines", line=dict(color="grey", dash="dash")), row=2, col=1)

        fig.add_trace(go.Scatter(x=np.diag(H), y=residual_std, mode="markers", marker=dict(color="royalblue")), row=2, col=2)
        fig.add_trace(go.Scatter(x=res_fit_3[:,0], y=res_fit_3[:,1], mode="lines", line=dict(color="red", dash="dash")), row=2, col=2)

        # plot high leverage points
        fig.add_trace(go.Scatter(x=np.diag(H)[high_leverage_points], y=residual_std[high_leverage_points], name="High Leverage Points", 
                                mode="markers+text", marker=dict(symbol="circle-open", color="black", size=12), 
                                text=list(map(str, list(np.argwhere(high_leverage_points).ravel()))), textposition="bottom center"), row=2, col=2)
        fig.add_trace(go.Scatter(x=[-0.05, np.diag(H).max()*1.2], y=[0,0], mode="lines", line=dict(color="grey", dash="dash", width=1)), row=2, col=2)
        fig.add_trace(go.Scatter(x=[0, 0], y=[residual_std.min(), residual_std.max()*1.2], mode="lines", line=dict(color="grey", dash="dash", width=1)), row=2, col=2)
        # plot Cooks distance 
        fig.add_trace(go.Scatter(x=np.arange(len(fitted)), y=CD, name="Cook's distance", mode="markers", marker=dict(symbol="x", color="royalblue")), row=3, col=1)
        fig.add_trace(go.Scatter(x=(np.arange(len(fitted)))[high_influence_points], y=CD[high_influence_points], mode="markers+text", 
                                marker=dict(symbol="circle-open", color="black", size=12), 
                                text=list(map(str, np.argwhere(high_influence_points).ravel())), textposition="top center"), row=3, col=1)

        
        fig.update_xaxes(title_text="Fitted values", row=1, col=1)
        fig.update_xaxes(title_text="Fitted values", row=1, col=2)
        fig.update_xaxes(title_text="Theoretical quantiles", row=2, col=1)
        fig.update_xaxes(title_text="Leverage", row=2, col=2,range=[-0.05, np.diag(H).max()*1.2])
        fig.update_xaxes(title_text="Obs. number", row=3, col=1)

        fig.update_yaxes(title_text="Residual", row=1, col=1)
        fig.update_yaxes(title_text="sqrt(standardized residuals)", row=1, col=2)
        fig.update_yaxes(title_text="Standardized residual", row=2, col=1)
        fig.update_yaxes(title_text="Standardized residual", row=2, col=2, range=[1.2*residual_std.min(), 1.2*residual_std.max()])
        fig.update_yaxes(title_text="Cook's distance", row=3, col=1, range=[-0., 1.2*CD.max()])
        
        fig.update_layout(height=1000, width=1000, title_text="Diagnostic Plots", showlegend=False)
        
        if fname:
            fig.write_image(fname+".pdf")
        fig.show()