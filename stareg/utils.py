# coding: utf-8

import numpy as np
from scipy.sparse import diags
from functools import singledispatch


@singledispatch
def mm(n_param : int, constraint="inc", dim=0):
    """Creates the mapping matrix for the constraint P-splines as in
    Fahrmeir, Regression p.436f, for the constraint.

    Paramters:
    ----------
    n_param : int     - Number of used B-spline basis functions.
    constraint : str  - Type of constraint.

    Returns:
    --------
    D : matrix     - Finite difference matrix of shape (n_param-order x n_param)
    """
    order = 1 if constraint in ["inc", "dec", "peak", "valley"] else 2
    assert (n_param > order), "n_param needs to be larger than order!"
    if order == 1:
        d1 = np.array([-1*np.ones(n_param),np.ones(n_param)])
        D = diags(d1,offsets=[0,1], shape=(n_param-order, n_param)).toarray()
    elif order == 2:
        d2 = np.array([np.ones(n_param),-2*np.ones(n_param),np.ones(n_param)])
        D = diags(d2,offsets=[0,1,2], shape=(n_param-order, n_param)).toarray()

    if constraint == "none":
        D = np.zeros((n_param, n_param))

    return D

@mm.register
def _(n_param : tuple, constraint="inc", dim=0):
    """Creates the mapping matrix for the constraint tensor-product P-splines 
    as in Fahrmeir, Regression p.508 equation (8.27) for the constraint.

    Paramters:
    ----------
    n_param : tuple     - Numbers of used B-spline basis functions.
    constraint : str    - Type of constraint.
    dim : int           - Indicator for the dimension of the constraint, 
                          0 for dimension 1, 1 for dimension 2, e.g. 
                          (10, "inc", 1) means 10 basis functions with increasing constraint
                          in dimension 2 for the two-dimensional data X = [x_1, x_2]

    Returns:
    --------
    D : matrix           - Mapping matrix for the constraint and dimension.
    """
    order = 1 if constraint in ["inc", "dec", "peak", "valley"] else 2

    assert (dim in [0, 1]), "Argument 'dim' either 0 or 1."
    assert (n_param[0] > order and n_param[1] > order), "n_param needs to be larger than order of constraint!"

    if order == 1:
        d = np.array([-1*np.ones(n_param[dim]),np.ones(n_param[dim])])
        D = diags(d,offsets=[0,1], shape=(n_param[dim]-order, n_param[dim])).toarray()
    elif order == 2:
        d = np.array([np.ones(n_param[dim]),-2*np.ones(n_param[dim]),np.ones(n_param[dim])])
        D = diags(d,offsets=[0,1,2], shape=(n_param[dim]-order, n_param[dim])).toarray()
    
    if dim == 0:
        Dc = np.kron(np.eye(n_param[dim+1]), D)
    else:
        Dc = np.kron(D, np.eye(n_param[dim-1]))

    if constraint == "none":
        Dc = np.zeros((np.prod(n_param), np.prod(n_param)))

    return Dc

def check_constraint(coef, constraint="inc", y=None, B=None):
    """Check whether the coefficients in coef hold true to the constraint for
    the B-spline coefficients.

    Parameters:
    -----------
    coef  : array     - Array of coefficients to test against the constraint.
    constraint : str  - Constraint type.
    y  : array        - Target data.
    B  : matrix       - B-spline basis matrix.

    Returns:
    --------
    v  : array      - Diagonal elements of the weighting matrix V.
    """

    threshold = 1e-4
    if constraint not in ["inc", "dec", "conv","conc", "peak", "valley", "none"]:
        print(f"Constraint '{constraint}'' currently not available.")
        return

    if constraint == "inc":
        v = np.diff(coef) < threshold
    elif constraint == "dec":
        v = np.diff(coef) > -threshold
    elif constraint == "conv":
        v = np.diff(coef, 2) < threshold
    elif constraint == "conc":
        v = np.diff(coef, 2) > -threshold
    elif constraint == "peak":
        assert (np.all(y != None) and np.all(B != None)), "Include the output y and B-spline basis matrix B."
        peakidx = np.argmax(y)
        peak_spline_idx = np.argmax(B[peakidx,:])
        v = list(np.diff(coef[:peak_spline_idx]) < threshold) + [0] + list(np.diff(coef[peak_spline_idx:]) > -threshold)
        v = np.array(v)
    elif constraint == "valley":
        assert (np.all(y != None) and np.all(B != None)), "Include the output y and B-spline basis matrix B."
        valleyidx = np.argmin(y)
        valley_spline_idx = np.argmax(B[valleyidx,:])
        v = list(np.diff(coef[:valley_spline_idx]) > -threshold) + [0] + list(np.diff(coef[valley_spline_idx:]) < threshold)
        v = np.array(v)
    else:
        v = np.zeros(len(coef))
    return v.astype(int)

def check_constraint_full_model(model, coef=None, basis=0, y=0):
    """ Tests the whole model against all constraints given in the model.
    
    Parameters:
    ----------
    model : dict    - Model dictionary, output from stareg.create_model_from_description.
    coef : array    - If None, uses coef_pls as test coefficients.
    basis : matrix  - Basis matrix to evaluate peak/valley constraint.
    y : arra        - Target data to evaluate peak/valley constraint
    Returns:
    --------
    W : dict        - Keys are the submodels, values are the weight vectors of the submodel.
    model : dict    - Updated model, the weights are changed to be consisted with W.
    
    """
    W = dict()
    coef_idx = 0
    for submodel in model.keys():
        type_ = model[submodel]["type"]
        len_submodel = len(model[submodel]["coef_pls"])
        if coef is not None:
            test_coef = coef[coef_idx:len_submodel+coef_idx]
            #ic(test_coef.shape)
        else:
            test_coef = model[submodel]["coef_pls"]
        test_constraints = model[submodel]["constraint"]
        #ic(test_constraints)
        if type_.startswith("s"):
            v = list(check_constraint(test_coef, constraint=test_constraints, y=y, B=model[submodel]["B"]))
            
        elif type_.startswith("t"):
            v1 = list(check_constraint_dim1(test_coef, test_constraints[0], nr_splines=model[submodel]["nr_splines"]))
            v2 = list(check_constraint_dim2(test_coef, test_constraints[1], nr_splines=model[submodel]["nr_splines"]))
            v = dict(v1=v1, v2=v2)
        
        model[submodel]["weights"] = v
        W[submodel] = v
        coef_idx += len_submodel
        #ic(coef_idx)
    return W, model


def check_constraint_full(coef_, descr, basis=0, y=0):
    """Checks the respective parts of the coef vector against 
    the respective constraints. 
    
    Paramters:
    ----------
    coef_ : array    - Vector of coefficients.
    descr : tuple    - Model description.
    
    Returns:
    --------
    v : list    - Diagonal elements of the weighting matrix V.
    vc : list   - Comparable version of the weighting matrix V.
    """
    i, v, vc = 0, [], []
    for e in descr:
        type_, nr_splines, constraints = e[0], e[1], e[2]
        if type_.startswith("s"):
            c = coef_[i:int(nr_splines)+i]
            vc += list(check_constraint(coef=c, constraint=constraints, y=y, B=basis[:, i:int(nr_splines)+i]))
            v.append(check_constraint(coef=c, constraint=constraints, y=y, B=basis[:, i:int(nr_splines)+i]))
        elif type_.startswith("t"):
            c = coef_[i:np.prod(nr_splines)+i]
            v1 = check_constraint_dim1(coef=c, nr_splines=nr_splines, constraint=constraints[0])
            v2 = check_constraint_dim2(coef=c, nr_splines=nr_splines, constraint=constraints[1])
            vc += list(v1) 
            vc += list(v2)
            v.append((v1,v2))
        else:
            print("Only B-splines (s) and tensor-product B-splines (t) are supported!")
            return
        i += np.prod(e[1])
    return v, vc



def check_constraint_dim2(coef, constraint="inc", nr_splines=(6,4)):
    """Compute the diagonal elements of the weighting matrix for SC-TP-P-splines 
    Compute the diagonal elements of the weighting matrix for SC-TP-P-splines 
    given the constraint for direction 2.
      
    According to the scheme given in the Master Thesis !!
    
    Parameters:
    -----------
    coef  : array      - Coefficient vector to test against constraint.
    constraint : str   - Specifies the constraint.
    nr_splines : list  - Specifies the number of splines in each dimension
    
    Returns
    -------
    v  : array         - Diagonal elements of the weighting matrix V.
    """
    if constraint in ["inc", "dec"]:
        diff = 1
    else:
        diff = 0
        
    v2 = np.zeros(nr_splines[0]*(nr_splines[1]-diff))
    for i in range(1, nr_splines[1]):
        for j in range(1, nr_splines[0]+1):
            # print(j+(i-1)*nr_splines[0]-1, "->", j+i*nr_splines[0]-1, "-", j+i*nr_splines[0]-nr_splines[0]-1)
            v2[j+(i-1)*nr_splines[0]-1] = coef[j+i*nr_splines[0]-1] - coef[j+i*nr_splines[0]-nr_splines[0]-1]
    if constraint == "inc":
        v2 = v2 < 0
    elif constraint == "dec":
        v2 = v2 > 0
    elif constraint == "none":
        v2 = np.zeros(v2.shape)
    
    return v2.astype(int)

def check_constraint_dim1(coef, constraint="inc", nr_splines=(6,4)):
    """Compute the diagonal elements of the weighting matrix for SC-TP-P-splines 
    Compute the diagonal elements of the weighting matrix for SC-TP-P-splines 
    given the constraint for direction 1.
      
    According to the scheme given in the Master Thesis !!
    
    Parameters:
    -----------
    coef  : array      - Coefficient vector to test against constraint.
    constraint : str   - Specifies the constraint.
    nr_splines : list  - Specifies the number of splines in each dimension
    
    Returns
    -------
    v  : array         - Diagonal elements of the weighting matrix V.
    """
    if constraint in ["inc", "dec"]:
        diff = 1
    else:
        diff = 0
    # first constraint in dim 1
    v1 = np.zeros((nr_splines[0]-diff)*nr_splines[1])
    for i in range(1,nr_splines[1]+1): 
        for j in range(nr_splines[0]-1):
            # print(j+(i-1)*(nr_splines[0]-1), "->", j+(i-1)*nr_splines[0] + 1, "-", j+(i-1)*nr_splines[0])
            v1[j+(i-1)*(nr_splines[0]-1)] = coef[j+(i-1)*nr_splines[0] + 1] - coef[j+(i-1)*nr_splines[0]]
            
    if constraint == "inc":
        v1 = v1 < 0
    elif constraint == "dec":
        v1 = v1 > 0
    elif constraint == "none":
        v1 = np.zeros(v1.shape)
    
    return v1.astype(int)