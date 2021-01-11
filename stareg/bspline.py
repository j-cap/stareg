# coding: utf-8

import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from utils import *
from functools import singledispatch

class Bspline():
    
    def basisfunction(self, X, knots, j, l):
        """B-spline basis function definition according to Fahrmeir, Regression p.429.

        Parameters:
        -----------
        X :      array   - Input data of shape (n_samples, ) to evaluate the B-spline basis function 
                           of order l on.
        knots :  array   - Knot sequence defining the B-spline basis function.
        j :      int     - Index of the B-spline basis function to evaluate.
        l :      int     - Order of the B-spline basis function, e.g. l=3 -> cubic.

        Returns:
        --------
        b : array     - B-spline basis function evaluated at X. 

        """
        if l == 0:
            b = ((knots[j] <= X) & (X < knots[j+1])).astype(int)
            return b
        else:
            b0 = (X - knots[j-l]) / (knots[j] - knots[j-l])
            b1 = (knots[j+1] - X) / (knots[j+1] - knots[j+1-l])
            b = b0*self.basisfunction(X, knots, j-1, l-1) + b1*self.basisfunction(X, knots, j, l-1)
            return b

    @classmethod
    def basismatrix(self, X, nr_splines=10, l=3, knot_type="e"):
        """Generate the B-spline basis matrix for nr_splines given the data X.

         Note: (nr_splines + l + 1) knots are needed for a B-spline basis of 
               order l with nr_splines, e.g. for l=3, nr_splines=10 -> len(knots) = 14. 

         Parameters:
         ----------
         X :          array  -  Input data of shape (n_samples, ) to compute the B-spline basis matrix for.
         nr_splines : int    -  Number of parameters (== number of B-spline basis functions).
         l :          int    -  Specifies the order of the B-spline basis functions.
         knot_type :  str    -  Decide between equidistant "e" and quantile based "q"
                                knot placement.

         Returns:
         --------
         B : matrix  - B-spline basis matrix of shape: ( length(x) x nr_splines ).
         k : array   - Knot sequence.

        """
        Bs = Bspline()
        B = np.zeros((len(X), nr_splines))
        xmin, xmax = X.min(), X.max()

        if knot_type is "e":
            knots_inner = np.linspace(xmin, xmax, nr_splines-l+1)
        elif knot_type is "q":
            p = np.linspace(0, 1, nr_splines-l+1);
            xs = np.sort(X, kind="quicksort")
            quantile_idx = np.array((len(X)-1)*p, dtype=np.int16)
            knots_inner = xs[quantile_idx]
        else:
            print(f"Knot Type {knot_type} not implemented!")
        
        dknots = np.diff(knots_inner).mean()
        knots_left = np.linspace(xmin-l*dknots, xmin-dknots, l)
        knots_right = np.linspace(xmax+dknots, xmax+l*dknots, l)
        knots = np.concatenate((knots_left, knots_inner, knots_right))

        for j in range(l,len(knots)-1):
            B[:,j-l] = Bs.basisfunction(X=X, knots=knots, j=j, l=l);

        return dict(basis=B, knots=knots)

    @classmethod
    def tensorproduct_basismatrix(self, X, nr_splines=(7,7), l=(3,3), knot_type=("e", "e")):
        """Generate the 2-d tensor-product B-spline basis matrix for nr_splines[0] and
        nr_splines[1] for dimension 1 and 2.

        Parameters
        ----------
        X : array          - Input data of shape (n_samples, 2)
        nr_splines : list  - Contains the number of B-spline basis functions for each dimension.
        l : list           - Spline order for each dimensions.
        knot_type : list   - Knot types for each dimension.

        Returns:
        --------
        T  : matrix   - Tensor-product B-spline basis.
        k1 : array    - Knot sequence of dimension 1.
        k2 : array    - Knot sequence of dimension 2.
        """
        BS = Bspline()
        B1, k1 = BS.basismatrix(X=X[:,0], nr_splines=nr_splines[0], l=l[0], knot_type=knot_type[0]).values()
        B2, k2 = BS.basismatrix(X=X[:,1], nr_splines=nr_splines[1], l=l[1], knot_type=knot_type[1]).values()

        n_samples, n_dim = X.shape
        T = np.zeros((n_samples, nr_splines[0]*nr_splines[1]))
        for i in range(n_samples):
            T[i,:] = np.kron(B2[i,:], B1[i,:])
            
        return dict(basis=T, knots1=k1, knots2=k2)
    
    @classmethod
    def fit(self, X, y, nr_splines=10, l=3, knot_type="e"):
        """Calculate the least squares parameters of the B-spline given the data X.

        Parameters:
        -----------
        X : array        - Input data of shape (n_samples, ).
        y : array        - Output data of shape (n_samples, ).
        nr_splines : int - Nr. of B-spline basis functions to use.
        l : int          - Order of the B-spline basis functions.
        knot_type : str  - Specifies the type of the knot sequence, either "e" or "q".

        Returns:
        --------
        coef_ : array  - Estimated coefficients of shape (nr_splines, ).
        B : matrix     - B-spline basis matrix.
        k : array      - Knot sequence 
        """
        if len(X.shape) == 1:
            B, k = self.basismatrix(X=X, nr_splines=nr_splines, l=l, knot_type=knot_type).values()
        elif X.shape[1] == 2:
            B, k1, k2 = self.tensorproduct_basismatrix(X=X, nr_splines=nr_splines, l=l, knot_type=knot_type).values()
            k = (k1, k2)
        else:
            print("Maximal dimension == 2!")
            return 
        solution = np.linalg.lstsq(a=B, b=y, rcond=None)      
        return dict(coef_=solution[0], basis=B, knots=k) 
    
    def predict(self, Xpred, coef, knots, l=3):
        """Calculate the B-spline value for X given the parameters in coef.

        Paramters:
        ----------
        X : array      - Data of shape (n_samples, ) to evaluate the B-spline on.
        coef  : array  - Parameters of the B-spline.
        knots : array  - Knot sequence of the B-spline.
        l : int        - Order of the B-spline.

        Returns:
        --------
        s : array   - B-spline values on X for the given parameters.

        """
        if len(Xpred.shape) == 1:
            print("Prediction for 1-D Data".center(30, "-"))
            B = np.zeros((len(Xpred), len(coef)))
            for j in range(l, len(knots)-1):
                B[:,j-l] = self.basisfunction(Xpred, knots, j, l)
                
        elif Xpred.shape[1] == 2:
            print("Prediction for 2-D Data".center(30, "-"))
            n_samples = len(Xpred[:,0])
            B1pred, B2pred = np.zeros((n_samples, len(knots[0])-1-l[0])), np.zeros((n_samples, len(knots[1])-1-l[1]))
            B = np.zeros((n_samples, len(coef)))

            for j in range(l[0], len(knots[0])-1):
                B1pred[:,j-l[0]] = self.basisfunction(Xpred[:,0], knots[0], j, l[0]) 
            for j in range(l[1], len(knots[1])-1):
                B2pred[:,j-l[1]] = self.basisfunction(Xpred[:,1], knots[1], j, l[1]) 
            for i in range(n_samples):
                B[i,:] = np.kron(B2pred[i,:], B1pred[i,:])    
        else:
            print("Maximal dimension == 2!")
            return 
        s = B @ coef
        return s    

    @classmethod
    def fit_Pspline(self, X, y, nr_splines=10, l=3, knot_type="e", lam=1):
        """Implementation of the P-spline functionality given in Fahrmeir, Regression p.431ff.

        Solves the Ridge regression style problem of the form:
            coef = (X^t X + lam D_2^t D_2)^{-1} (X^t y)

        using X   ... B-spline basis matrix.
              D_2 ... Second-order finite difference matrix.
              lam ... Smoothness parameter. 

        Parameters:
        -----------
        X : array        - Input data of shape (n_samples, ).
        y : array        - Output data of shape (n_samples, ).
        nr_splines : int - Nr. of B-spline basis functions to use.
        l : int          - Order of the B-spline basis functions.
        knot_type : str  - Specifies the type of the knot sequence, either "e" or "q"
        lam : float      - Value of the smoothness parameter.

        Returns:
        --------
        coef_ : array  - Estimated coefficients of shape (nr_splines, ).
        B : matrix     - B-spline basis matrix.
        k : array      - Knot sequence 
        """
        if len(X.shape) == 1:
            B, k = self.basismatrix(X=X, nr_splines=nr_splines, l=l, knot_type=knot_type).values()
            D2 = mm(nr_splines, constraint="smooth")
            coef_ = np.linalg.pinv(B.T@B + lam * (D2.T@D2)) @ (B.T @ y)
            D = D2
        elif X.shape[1] == 2:
            T, k1, k2 = self.tensorproduct_basismatrix(X=X, nr_splines=nr_splines, l=l, knot_type=knot_type).values()
            k = (k1, k2)
            D1 = mm(nr_splines, constraint="smooth", dim=0)
            D2 = mm(nr_splines, constraint="smooth", dim=1)
            coef_ = np.linalg.pinv(T.T@T + lam * (D1.T@D1) + lam * (D2.T@D2)) @ (T.T@y)
            D, B = (D1, D2), T
        else:
            print("Maximal dimension == 2!")
            return 

        self.Pspline_coef_ = coef_
        return dict(coef_=coef_, basis=B, knots=k, mapping_matrices=D)

    def calc_GCV(self, X, y, nr_splines=10, l=3, knot_type="e", nr_lam=10, plot_=1):
        """Calculate the generalized cross validation for the given data (x,y).

        Parameters: 
        -----------
        X : array        - Input data of shape (n_samples, 1).
        y : array        - Output data of shape (n_samples, ).
        nr_splines : int - Nr. of B-spline basis functions to use.
        l : int          - Order of the B-spline basis functions.
        knot_type : str  - Specifies the type of the knot sequence, either "e" or "q"
        nr_lam : float   - Number of lambdas to try out in the GCV.
        plot_    : bool  - Plot fit and smoothing paramter curve.

        Returns:
        --------
        coef_    : array   - Optimal coefficients for penalized least squares
                             fit using generalized cross validation.
        B        : matrix  - B-spline basis matrix.
        k        : array   - Knot sequence.
        best_lam : float   - Optimal smoothing parameter.
        """

        lambdas = np.logspace(-8,8,num=nr_lam)
        gcvs = np.zeros(nr_lam)
        B,k = self.basismatrix(X=X, nr_splines=nr_splines, l=l, knot_type=knot_type).values()
        D = mm(nr_splines, constraint="smooth", dim=0)
        BtB, DtD = B.T@B, D.T@D

        for i, lam in enumerate(tqdm(lambdas)):
            coef_pls = np.linalg.pinv(BtB + lam*DtD) @ (B.T @ y)
            trace_H = np.trace((BtB) @ np.linalg.pinv(BtB + lam * (DtD)))
            ypred = B @ coef_pls
            gcvs[i] = sum(((y - ypred) / (1 - trace_H / len(y)) )**2) / len(y)

        best_gcv_idx = np.argmin(gcvs)
        best_lam = lambdas[best_gcv_idx]
        if plot_:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lambdas, y=gcvs, mode="lines", line=dict(width=1, color="blue"), name="Lambdas"))
            fig.add_trace(go.Scatter(x=[lambdas[best_gcv_idx]], y=[gcvs[best_gcv_idx]], mode="markers", 
                                     marker=dict(size=10, color="red", symbol=100), name="Optimal Value"))
            fig.update_layout(title="GCV-Search")        
            fig.update_xaxes(title_text = "lambdas", type="log")
            fig.update_yaxes(title_text = "GCV-Score", type="log")
            fig.show()

        return dict(best_lambda=best_lam)
    

    def calc_GCV_2d(self, X, y, nr_splines=(10,10), l=(3,3), knot_type=("e","e"), nr_lam=10, plot_=1, verbose=0):

        lam_test = np.logspace(-4,4, nr_lam)
        gcvs = np.zeros(len(lam_test))

        T, k1, k2 = self.tensorproduct_basismatrix(X, nr_splines).values()
        D1 = mm(nr_splines, constraint="smooth", dim=0)
        D2 = mm(nr_splines, constraint="smooth", dim=1)
        TtT = T.T@T
        D1tD1, D2tD2 = D1.T@D1, D2.T@D2

        for i, lam in enumerate(tqdm(lam_test)):
            coef_pls = np.linalg.pinv(TtT + lam*D1tD1 + lam*D2tD2) @ (T.T @ y)
            trace_H = np.trace((TtT) @ np.linalg.pinv(TtT + lam*D1tD1 + lam*D2tD2))
            ypred = T @ coef_pls
            gcvs[i] = sum(((y - ypred) / (1 - trace_H / len(y)) )**2) / len(y)

        best_gcv_idx = np.argmin(gcvs)
        best_lam = lam_test[best_gcv_idx]

        if plot_:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lam_test, y=gcvs, mode="lines", line=dict(width=1, color="blue"), name="Lambdas"))
            fig.add_trace(go.Scatter(x=[lam_test[best_gcv_idx]], y=[gcvs[best_gcv_idx]], mode="markers", 
                                     marker=dict(size=10, color="red", symbol=100), name="Optimal Value"))
            fig.update_layout(title="GCV-Search")    
            fig.update_xaxes(title_text = "Lambda 1", type="log")
            fig.show()  
        return dict(best_lambda=best_lam) #, best_lambda2=best_lam2)
    
    
    # legacy code
    def fit_SC_Pspline(self, X, y, constraint="none", nr_splines=10, l=3, knot_type="e", lam_c=6000):
        """Implementation of the shape-constraint P-spline fit.

        Solve the Ridge regression style problem of the form:
            coef = (X^t X + lam D_2^t D_2 + lam_c D_c^t V D_c)^{-1} (X^t y)

        using X   ... B-spline basis matrix.
              D_2 ... Second-order finite difference matrix.
              lam ... Smoothness parameter. 
              D_c ... Mapping matrix of the constraint.
              V   ... Weighting matrix of the constraint.
              lam_c ... Constraint parameter.
        with lam give as optimal smoothing parameter by GCV.
        
        Parameters:
        -----------
        X : array        - Input data of shape (n_samples, ).
        y : array        - Output data of shape (n_samples, ).
        nr_splines : int - Nr. of B-spline basis functions to use.
        l : int          - Order of the B-spline basis functions.
        knot_type : str  - Specifies the type of the knot sequence, either "e" or "q"
        lam_c : float    - Value of the constraint parameter.
        constraint : str - Type of constraint.

        Returns:
        --------
        coef_ : array  - Estimated coefficients of shape (nr_splines, ).
        B : matrix     - B-spline basis matrix.
        k : array      - Knot sequence 
        lams : list    - Used smoothness parameter and and constraint parameter. 
        """

        lam = self.calc_GCV(X=X, y=y, nr_splines=nr_splines, l=l, knot_type=knot_type, nr_lam=100, plot_=0)["best_lambda"]
        coef_pls, B, k, D = self.fit_Pspline(X=X, y=y,  nr_splines=nr_splines, l=l, knot_type=knot_type, lam=lam).values()

        Ds = mm(nr_splines, constraint="smooth")
        Dc = mm(nr_splines, constraint=constraint)
        v = check_constraint(coef=coef_pls, constraint=constraint, y=y, B=B)
        vold = [0]*len(v)

        BtB = B.T @ B
        Bty = B.T @ y
        DstDs = Ds.T@Ds
        i = 1
        print(f"Pre-Iteration".center(30, "="))
        print(f"MSE = {mean_squared_error(y, B @coef_pls).round(7)}".center(30, "-"))
        df = pd.DataFrame(data=coef_pls)

        while not (list(v) == list(vold)):
            vold = v
            coef_pls = np.linalg.pinv(BtB + lam*DstDs + lam_c * Dc.T @ np.diag(v) @ Dc) @ (Bty)
            v = check_constraint(coef=coef_pls, constraint=constraint, y=y, B=B)
            print(f" Iteration {i} ".center(30, "="))
            print(f"MSE = {mean_squared_error(y, B @coef_pls).round(7)}".center(30, "-"))
            i += 1
            df = pd.concat([df, pd.DataFrame(data=coef_pls)], axis=1)

        return dict(coef_=coef_pls, basis=B, knots=k, lambdas=(lam, lam_c), df_coef=df)

    # legacy code
    def fit_SC_TP_Pspline(self, X, y, constraints=("none", "none"), nr_splines=(10, 10), l=(3,3), knot_type=("e","e"), lam_c=(6000, 6000)):
        """Implementation of the constrained tensor-product P-spline fit.

        Solve the Ridge regression style problem of the form:
            coef = (X^t X + lam D_2^t D_2 + lam_c1 D_c1^t V1 D_c1 + lam_c2 D_C2^t V2 D_c2)^{-1} (X^t y)

        using X   ... B-spline basis matrix.
              D_2 ... Second-order finite difference matrix.
              lam ... Smoothness parameter. 
              D_c12 ... Mapping matrix of the constraint 1.
              D_c1  ... Mapping amtrix fo constraint 2.
              V1    ... Weighting matrix of the constraint 1.
              V2    ... Weighting matrix of the constraint 1.
              lam_c ... Constraint parameter.
        with lam give as optimal smoothing parameter by GCV.

        Parameters:
        -----------
        X : array        - Input data of shape (n_samples, ).
        y : array        - Output data of shape (n_samples, ).
        nr_splines : int - Nr. of B-spline basis functions to use.
        l : int          - Orders of the B-spline basis functions.
        knot_type : str  - Specifies the types of the knot sequence, either "e" or "q"
        lam_c : tuple    - Values of the constraint parameters.

        Returns:
        --------
        coef_ : array  - Estimated coefficients of shape (nr_splines, ).
        B : matrix     - B-spline basis matrix.
        k : array      - Knot sequence 
        lams : list    - Used smoothness parameter and and constraint parameter. 
        df : DataFrame - Contains the estimated coefficients for all interations.
        """

        lam = self.calc_GCV_2d(X, y, nr_splines=nr_splines, l=l, knot_type=knot_type, plot_=1, nr_lam=100)["best_lambda"]   
        coef_pls, B, k, D = self.fit_Pspline(X=X, y=y,  nr_splines=nr_splines, l=l, knot_type=knot_type, lam=lam).values()

        D1 = mm(nr_splines, constraint="smooth", dim=0)
        D2 = mm(nr_splines, constraint="smooth", dim=1)
        DC1 = mm(nr_splines, constraint=constraints[0], dim=0)
        DC2 = mm(nr_splines, constraint=constraints[1], dim=1)
        v1 = check_constraint_dim1(coef=coef_pls, nr_splines=nr_splines, constraint=constraints[0])
        v2 = check_constraint_dim2(coef=coef_pls, nr_splines=nr_splines, constraint=constraints[1])

        v1old = [1]*len(v1)
        v2old = [1]*len(v2)

        BtB = B.T @ B
        Bty = B.T @ y
        DstDs_1 = D1.T@D1
        DstDs_2 = D2.T@D2

        i = 1
        df = pd.DataFrame(data=coef_pls)
        print(f"unconstrained MSE = {mean_squared_error(y, B @coef_pls).round(7)}".center(30, "-"))
        while not ((list(v1) == list(v1old)) and (list(v2) == list(v2old))):
            v1old = v1
            v2old = v2
            Dc1tV1Dc1 = DC1.T @ np.diag(v1) @ DC1
            Dc2tV1Dc2 = DC2.T @ np.diag(v2) @ DC2
            coef_pls = np.linalg.pinv(BtB + lam*DstDs_1 + lam*DstDs_2 + lam_c[0] * Dc1tV1Dc1 +  lam_c[1] * Dc2tV1Dc2) @ (Bty)
            v1 = check_constraint_dim1(coef=coef_pls, nr_splines=nr_splines, constraint=constraints[0])
            v2 = check_constraint_dim2(coef=coef_pls, nr_splines=nr_splines, constraint=constraints[1])
            print(f" Iteration {i} ".center(30, "="))
            print(f"MSE = {mean_squared_error(y, B @coef_pls).round(7)}".center(30, "-"))
            i += 1
            df = pd.concat([df, pd.DataFrame(data=coef_pls)], axis=1)

        return dict(coef_=coef_pls, basis=B, knots=k, lambdas=(lam, (lam_c[0], lam_c[1])), df_coef=df)
