# coding: utf-8

import numpy as np
from bspline import Bspline
from utils import *

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
        # weights = check_constraint_full(coef_=coef_pls, descr=m, basis=B, y=y)

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
        time.sleep(0.2)
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