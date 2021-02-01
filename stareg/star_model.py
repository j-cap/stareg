# coding: utf-8

import numpy as np
import pandas as pd
import time
from scipy.linalg import block_diag
from sklearn.metrics import mean_squared_error

from .bspline import Bspline
from .utils import mm, check_constraint, check_constraint_dim1
from .utils import check_constraint_dim2, check_constraint_full_model

class Stareg():
    
    def __init__(self):
        print("Class initialization")
        self.BS = Bspline()
        
    def fit(self,description, X, y):
        model = self.create_model_from_description(description, X, y)

        iterIdx = 1
        B = self.create_basis_matrix(model)
        S = self.create_smoothness_matrix(model)
        K = self.create_constraint_matrix(model)

        BtB = B.T @ B
        Bty = B.T @ y

        weights_compare, model = check_constraint_full_model(model)
        weights_old = dict()

        while not weights_compare == weights_old:
            weights_old = weights_compare
            coef_cpls = np.linalg.pinv(BtB + S + K) @ Bty
            weights_compare, model = check_constraint_full_model(model, coef=coef_cpls)

            print(f"Iteration {iterIdx}".center(50, "="))
            print(f"MSE: {mean_squared_error(y, B @ coef_cpls)}".center(50, " "))

            # need 
            K = self.create_constraint_matrix(model)
            time.sleep(1)
            iterIdx += 1
            if iterIdx > 15:
                print("Stop the count!")
                break

        print("".center(50, "="))
        print("Iteration Finished!".center(50,"-"))
        print("".center(50, "="))

        return dict(coef_=coef_cpls, B=B, S=S, K=K, model=model)
        
    def predict(self, Xpred, model, coef_):
        """Calculate the predictions for Xpred and the given model and coef_.
        
        Parameters:
        -----------
        Xpred : array    - Input data to calculate the predictions for.
        model : dict     - Model dictionary.
        coef_ : array    - Coefficients of the constraint B-splines.
        
        Returns:
        --------
        ypred : array     - Predicted values.
        """
        
        basis = []
        for submodel in model.keys():
            type_ = model[submodel]["type"]
            nr_splines = model[submodel]["nr_splines"]
            knot_types = model[submodel]["knot_type"]
            knots = model[submodel]["knots"]
            order = model[submodel]["order"]
            print("Process ", type_)
            #time.sleep(0.2)
            if type_.startswith("s"):
                dim = int(type_[2])-1
                data = Xpred[:,dim]
                B = np.zeros((len(data), nr_splines))
                for j in range(order, len(knots)-1):
                    B[:,j-order] = self.BS.basisfunction(data, knots, j, order).ravel()
            elif type_.startswith("t"):
                dim1, dim2 = int(type_[2])-1, int(type_[4])-1
                data = Xpred[:,[dim1, dim2]]
                n_samples = len(data[:,0])
                B1, B2 = np.zeros((n_samples, len(knots["k1"])-1-order[0])), np.zeros((n_samples, len(knots["k2"])-1-order[1]))
                B = np.zeros((n_samples, np.prod(nr_splines)))
                for j in range(order[0], len(knots["k1"])-1):
                    B1[:,j-order[0]] = self.BS.basisfunction(data[:,0], knots["k1"], j, order[0]) 
                for j in range(order[1], len(knots["k2"])-1):
                    B2[:,j-order[1]] = self.BS.basisfunction(data[:,1], knots["k2"], j, order[1]) 
                for i in range(n_samples):
                    B[i,:] = np.kron(B2[i,:], B1[i,:])    
            else:
                print("Only B-splines (s) and tensor-product B-splines (t) are supported!")
            basis.append(B)
        # create combined basis matrix
        B = np.concatenate(basis, axis=1)
        y = B @ coef_

        return y


        
    def create_model_from_description(self, description, X, y):
        model = dict()
        parameter = ("type", "nr_splines", "constraint", "lambda_c", "knot_type")
        for idx, submodel in enumerate(description):
            model[f"f{idx+1}"] = dict()
            for type_and_value in zip(parameter, submodel):
                model[f"f{idx+1}"][type_and_value[0]] = type_and_value[1]

        for submodel in model:
            for key in model[submodel].keys():
                if key == "type":
                    type_ = model[submodel][key]
                    nr_splines = model[submodel]["nr_splines"]
                    knot_type = model[submodel]["knot_type"]
                    constraint = model[submodel]["constraint"]
                    lambda_c = model[submodel]["lambda_c"]
                    if type_.startswith("s"):
                        dim = int(type_[2])-1
                        data = X[:,dim]
                        order = 3
                        B, knots = self.BS.basismatrix(X=data, nr_splines=nr_splines, l=order, knot_type=knot_type).values()
                        Ds = mm(nr_splines, constraint="smooth")
                        Dc = mm(nr_splines, constraint=constraint)
                        lam = self.BS.calc_GCV(data, y, nr_splines=nr_splines, l=order, knot_type=knot_type, nr_lam=100, plot_=0)["best_lambda"]
                        coef_pls = self.BS.fit_Pspline(data, y, nr_splines=nr_splines, l=order, knot_type=knot_type, lam=lam)["coef_"]
                        W = check_constraint(coef_pls, constraint, y=y, B=B)
                    elif type_.startswith("t"):
                        dim = [int(type_[2])-1,  int(type_[4])-1]
                        data = X[:,[dim[0],dim[1]]]
                        order = (3,3)
                        B, knots1, knots2 = self.BS.tensorproduct_basismatrix(X=data, nr_splines=nr_splines, l=order, knot_type=knot_type).values()
                        Ds1 = mm(nr_splines, constraint="smooth", dim=0)
                        Ds2 = mm(nr_splines, constraint="smooth", dim=1)
                        Dc1 = mm(nr_splines, constraint=constraint[0], dim=0)                
                        Dc2 = mm(nr_splines, constraint=constraint[1], dim=1)                
                        lam = self.BS.calc_GCV_2d(data, y, nr_splines=nr_splines, l=order, knot_type=knot_type, nr_lam=100, plot_=0)["best_lambda"]
                        coef_pls = self.BS.fit_Pspline(data, y, nr_splines=nr_splines, l=order, knot_type=knot_type, lam=lam)["coef_"]
                        W1 = check_constraint_dim1(coef_pls, constraint[0], nr_splines)
                        W2 = check_constraint_dim2(coef_pls, constraint[1], nr_splines)

                        knots, Ds, Dc, W = dict(), dict(), dict(), dict()
                        knots["k1"], knots["k2"] = knots1, knots2
                        Ds["Ds1"], Ds["Ds2"] = Ds1, Ds2
                        Dc["Dc1"], Dc["Dc2"] = Dc1, Dc2
                        W["v1"], W["v2"] = W1, W2

            model[submodel]["B"] = B
            model[submodel]["knots"] = knots
            model[submodel]["Ds"] = Ds
            model[submodel]["Dc"] = Dc
            model[submodel]["weights"] = W
            model[submodel]["coef_pls"] = coef_pls
            model[submodel]["best_lambda"] = lam
            model[submodel]["order"] = order

        return model

           
    def create_basis_matrix(self, model):
        B = []
        [B.append(model[submodel]["B"]) for submodel in model.keys()]
        basis_matrix = np.concatenate(B,axis=1)
        return basis_matrix
    
    def create_smoothness_matrix(self, model):
        Ds = []
        for submodel in model.keys():
            type_ = model[submodel]["type"]
            if type_.startswith("s"):
                Ds.append(model[submodel]["best_lambda"] * model[submodel]["Ds"].T @ model[submodel]["Ds"])
            elif type_.startswith("t"):
                Ds1 = model[submodel]["Ds"]["Ds1"]
                Ds2 = model[submodel]["Ds"]["Ds2"]
                Ds.append(model[submodel]["best_lambda"] * (Ds1.T@Ds1 + Ds2.T@Ds2))
        
        smoothness_matrix = block_diag(*Ds)
        return smoothness_matrix
        
    def create_constraint_matrix(self, model):
        Dc = []
        for submodel in model.keys():
            type_ = model[submodel]["type"]
            if type_.startswith("s"):
                Dc.append(model[submodel]["lambda_c"] * model[submodel]["Dc"].T @ np.diag(model[submodel]["weights"]) @ model[submodel]["Dc"])               
            elif type_.startswith("t"):
                Dc1 = model[submodel]["Dc"]["Dc1"]
                Dc2 = model[submodel]["Dc"]["Dc2"]
                weights1 = np.diag(model[submodel]["weights"]["v1"])
                weights2 = np.diag(model[submodel]["weights"]["v2"])
                Dc.append(model[submodel]["lambda_c"][0]*(Dc1.T@weights1@Dc1) + model[submodel]["lambda_c"][1]*(Dc2.T@weights2@Dc2))

        constraint_matrix = block_diag(*Dc)         
        return constraint_matrix
    
    def create_coef_vector(self, model):
        coef = []
        [list(coef.append(model[submodel]["coef_pls"])) for submodel in model.keys()]
        return coef
    
    def calc_edof(self, B, S, K):
        """Calculates the effective degree of freedom according to Fahrmeir, Regression 2013, p.475.
        
        Parameters:
        -----------
        B : matrix     - Basis matrix of the model.
        S : matrix     - Smoothness penalty matrix of the model, aka. lam_s * D_2.T @ D_2.
        K : matrix     - Constraint penalty matrix of the model, aka. lam_c * D_c.T @ D_c.
        
        Returns:
        --------
        edof : float   - Effective degree of freedom.
        """
        BtB = B.T @ B
        edof = np.trace(BtB @ np.linalg.pinv(BtB + S + K))
        return edof
        

""" LEGACY CODE STARTS HERE """


def star_model(descr, X, y):
    """Fit a structured additive regression model using B-splines to the data in (X,y). 
    
    Parameters:
    -----------
    descr : tuple of tuples    - Describes the model structure, e.g. 
                                 descr = ( ("s(1)", 100, "inc", 6000, "e"), 
                                         ("t(1,2)", (12,10), ("none", "none"), (6000,6000), ("e", "e")), ),
                                 describing a model using a P-spline with increasing constraint and 100 
                                 basis functions for dimension 1 and a tensor-product P-spline without 
                                 constraints using 12 and 10 basis functions for the respective dimension.
    X : array                  - np.array of the input data, shape (n_samples, n_dim)
    y : array                  - np.array of the target data, shape (n_samples, )
    
    Returns:
    --------
    d : dict      - Returns a dictionary with the following key-value pairs:
                        basis=B, 
                        smoothness=S, 
                        constraint=K, 
                        opt_lambdas=optimal_lambdas, 
                        coef_=coef_pls, 
                        weights=weights
    """
    
    BS, TS = Bspline(), Bspline()
    coefs = []
    basis, smoothness = [], []
    constr, optimal_lambdas = [], []
    weights, weights_compare = [], []
    S, K = [], []

    for e in descr:
        type_, nr_splines, constraints, lam_c, knot_types = e[0], e[1], e[2], e[3], e[4]
        print("Process ", type_)
        if type_.startswith("s"):
            dim = int(type_[2])
            B = BS.basismatrix(X=X[:,dim-1], nr_splines=nr_splines, l=3, knot_type=knot_types)["basis"]
            Ds = mm(nr_splines, constraint="smooth", dim=dim-1)
            Dc = mm(nr_splines, constraint=constraints, dim=dim-1)
            lam = BS.calc_GCV(X=X[:,dim-1], y=y, nr_splines=nr_splines, l=3, knot_type=knot_types, nr_lam=50)["best_lambda"]
            coef_pls = BS.fit_Pspline(X=X[:,dim-1], y=y, nr_splines=nr_splines, l=3, knot_type=knot_types, lam=lam)["coef_"]
            W = check_constraint(coef=coef_pls, constraint=constraints, y=y, B=B)
            weights.append(W)
            weights_compare += list(W)
        elif type_.startswith("t"):
            print("Constraint = ", constraints)
            dim1, dim2 = int(type_[2]), int(type_[4])
            B = TS.tensorproduct_basismatrix(X=X[:,[dim1-1, dim2-1]], nr_splines=nr_splines, l=(3,3), knot_type=knot_types)["basis"]
            Ds1 = mm(nr_splines, constraint="smooth", dim=dim1-1)
            Ds2 = mm(nr_splines, constraint="smooth", dim=dim2-1)
            Dc1 = mm(nr_splines, constraint=constraints[0], dim=dim1-1)
            Dc2 = mm(nr_splines, constraint=constraints[1], dim=dim2-1)
            lam = TS.calc_GCV_2d(X=X[:,[dim1-1, dim2-1]], y=y, nr_splines=nr_splines, l=(3,3), knot_type=knot_types, nr_lam=50)["best_lambda"]
            coef_pls = TS.fit_Pspline(X=X[:,[dim1-1, dim2-1]], y=y, nr_splines=nr_splines, l=(3,3), knot_type=knot_types, lam=lam)["coef_"]
            W1 = check_constraint_dim1(coef_pls, nr_splines=nr_splines, constraint=constraints[0])
            W2 = check_constraint_dim2(coef_pls, nr_splines=nr_splines, constraint=constraints[1])
            weights.append((W1, W2))
            weights_compare += list(W1) + list(W2)
            Ds = (Ds1, Ds2)
            Dc = (Dc1, Dc2)
        else:
            print("Only B-splines (s) and tensor-product B-splines (t) are supported!")

        basis.append(B)
        smoothness.append(Ds)
        constr.append(Dc)
        optimal_lambdas.append(lam)
        coefs.append(coef_pls)

    coef_pls = np.concatenate(coefs)
    # create combined basis matrix
    B = np.concatenate(basis, axis=1)

    # create combined smoothness matrix
    for i, s in enumerate(smoothness):
        if len(s) == 2:
            S.append(optimal_lambdas[i]*(s[0].T @ s[0] + s[1].T@s[1]))
        else:
            S.append(optimal_lambdas[i]*(s.T@s))
    S = block_diag(*S)
    # create combined constraint matrix
    for i, c in enumerate(constr):
        if len(c) == 2:
            K.append(6000*(c[0].T @ np.diag(weights[i][0]) @ c[0]) + 6000*(c[1].T @np.diag(weights[i][1]) @ c[1]))
        else:
            K.append(6000* (c.T@ np.diag(weights[i])@c))
    K = block_diag(*K)

    weights_old = [0]*len(weights_compare)
    iterIdx = 1
    BtB = B.T @ B
    Bty = B.T @ y

    # Iterate till no change in weights
    df = pd.DataFrame(data=dict(w0=np.ones(2*12*10+100+100-2)))
    while not (weights_compare == weights_old):
        weights_old = weights_compare
        coef_pls = np.linalg.pinv(BtB + S + K) @ (Bty)
        weights, weights_compare = check_constraint_full(coef_=coef_pls, descr=descr, basis=B, y=y)

        print(f" Iteration {iterIdx} ".center(50, "="))
        print(f"MSE = {mean_squared_error(y, B@coef_pls).round(7)}".center(50, "-"))

        K = []
        print("Calculate new constraint matrix K".center(50,"-"))
        for i, c in enumerate(constr):
            if len(c) == 2:
                K.append(descr[i][3][0]*(c[0].T @ np.diag(weights[i][0]) @ c[0]) + descr[i][3][0]*(c[1].T @np.diag(weights[i][1]) @ c[1]))
            else:
                K.append(descr[i][3]*(c.T@ np.diag(weights[i])@c))
        K = block_diag(*K)

        df = pd.concat([df, pd.DataFrame(data={"w"+str(iterIdx):weights_compare})], axis=1)

        if iterIdx > 200:
            print("breaking")
            break

        iterIdx += 1
        
    print("Iteration Finished".center(50, "#"))
    return dict(basis=B, smoothness=S, constraint=K, opt_lambdas=optimal_lambdas, coef_=coef_pls, weights=weights)


def star_model_predict(Xpred, coefs, descr):
    BS, TS = Bspline(), Bspline()
    basis = []

    for e in descr:
        type_, nr_splines, constraints, lam_c, knot_types = e[0], e[1], e[2], e[3], e[4]
        print("Process ", type_)
        #time.sleep(0.2)
        if type_.startswith("s"):
            dim = int(type_[2])
            B = BS.basismatrix(X=Xpred[:,dim-1], nr_splines=nr_splines, l=3, knot_type=knot_types)["basis"]
        elif type_.startswith("t"):
            dim1, dim2 = int(type_[2]), int(type_[4])
            B = TS.tensorproduct_basismatrix(X=Xpred[:,[dim1-1, dim2-1]], nr_splines=nr_splines, l=(3,3), knot_type=knot_types)["basis"]
        else:
            print("Only B-splines (s) and tensor-product B-splines (t) are supported!")
        basis.append(B)
    
    # create combined basis matrix
    B = np.concatenate(basis, axis=1)
    y = B@coefs
    
    return dict(basis=B, y=y, X=Xpred)