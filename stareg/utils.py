#!/usr/bin/env python
# coding: utf-8

import copy
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

def check_constraint(beta, constraint, smooth_type="spline"):
    """Checks if array beta fits the constraint.
    
    Parameters
    ----------
    beta  : array
        Array of coefficients to be tested against the constraint.
    constraint : str
        Name of the constraint.
    smooth_type : str
        Either stareg.smooths.Smooths or stareg.smooths.TensorProductSmooths

    Returns
    -------
    V : np.ndarray
        Matrix with 1 where the constraint is violated, 0 else.
    
    """

    b_diff = np.diff(beta)
    b_diff_diff = np.diff(b_diff)
    if str(smooth_type) == "<class 'stareg.smooth.Smooths'>":
        if constraint == "inc":
            v = [0 if i > 0 else 1 for i in b_diff] 
        elif constraint == "dec":
            v = [0 if i < 0 else 1 for i in b_diff] 
        elif constraint == "conv":
            v = [0 if i > 0 else 1 for i in b_diff_diff] 
        elif constraint == "conc":
            v = [0 if i < 0 else 1 for i in b_diff_diff] 
        elif constraint == "smooth":
            v = list(np.ones(len(b_diff_diff), dtype=np.int))
        elif constraint == "peak":
            v = check_peak_constraint(beta=beta)
        elif constraint == "multi-peak":
            v = check_multi_peak_constraint(beta=beta)
        elif constraint == "valley":
            v = check_valley_constraint(beta=beta)
        elif constraint == "multi-valley":
            v = check_multi_valley_constraint(beta=beta)
        elif constraint == "peak-and-valley":
            v = check_peak_and_valley_constraint(beta=beta)

    elif str(smooth_type) == "<class 'stareg.smooth.TensorProductSmooths'>":
        if constraint == "inc":
            v = check_constraint_inc_tps(beta=beta)
        elif constraint == "dec":
            v = check_constraint_dec_tps(beta=beta)
        else:
            print(f"Constraint {constraint} not implemented for TPS")
            return

    return np.diag(v)

def check_constraint_inc_tps(beta):
    """Calculate the weight vector v for the increasing constraint for tps.

    Parameters
    ----------
    beta : array
        Array of coefficients to test for increasing constraint.

    Returns
    -------
    v : array
        Vector with 1 where constraint is violated, 0 elsewhere.

    """
    n_coef = int(np.sqrt(len(beta)))
    beta = beta.reshape(n_coef, n_coef)
    V1 = np.zeros((n_coef, n_coef))
    V2 = np.zeros((n_coef, n_coef))
    V1[:,1:] = np.diff(beta)
    V1[1:,0] = np.diff(beta[:,0])
    V2[1:,:] = np.diff(beta, axis=0)
    V2[0,1:] = np.diff(beta[0,:])
    v = np.logical_or(V1 < 0, V2 < 0).flatten()
    return v.astype(int)

def check_constraint_dec_tps(beta):
    """Calculate the weight vector v for the decreasing constraint for tps.

    Parameters
    ----------
    beta : array
        Array of coefficients to test for decreasing constraint.

    Returns
    -------
    v : array
        Vector with 1 where constraint is violated, 0 elsewhere.

    """
    n_coef = int(np.sqrt(len(beta)))
    beta = beta.reshape(n_coef, n_coef)
    V1 = np.zeros((n_coef, n_coef))
    V2 = np.zeros((n_coef, n_coef))
    V1[:,1:] = np.diff(beta)
    V1[1:,0] = np.diff(beta[:,0])
    V2[1:,:] = np.diff(beta, axis=0)
    V2[0,1:] = np.diff(beta[0,:])
    v = np.logical_or(V1 > 0, V2 > 0).flatten()
    return v.astype(int)

def check_valley_constraint(beta):
    """Calculate the weight vector v for valley constraint.

    Parameters
    ----------
    beta : array
        Array of coefficients to test for valley constraint.

    Returns
    -------
    v : array
        Vector with 1 where constraint is violated, 0 elsewhere.

    """

    idx = find_peaks(-beta, distance=len(beta))[0][0]
    left = np.diff(beta[:idx+1]) > 0
    right = np.diff(beta[idx:]) < 0
    v = np.array(list(left)+list(right))
    return v.astype(np.int)

def check_multi_valley_constraint(beta):
    """Check whether beta contains 2 peaks and is increasing to the first,
    then decreasing, then again increasing and then again decreasing

    Parameters
    ----------
    beta : array
        Array of coefficients to test for multi_valley constraint

    Returns
    -------
    v : array
        Vaector with 1 where constraint is violated, 0 elsewhere
    """

    valleys, _ = find_peaks(x=-1*beta, prominence=np.std(beta), distance=int(len(beta)/3))
    middle_spline = int(np.mean(valleys))
    v1 = check_valley_constraint(beta=beta[:middle_spline+1])
    v2 = check_valley_constraint(beta=beta[middle_spline:])
    v = np.array(list(v1)+list(v2))
    return v.astype(np.int)

def check_peak_constraint(beta):
    """Calculate the weight vector v for peak constraint.

    Parameters
    ----------
    beta : array
        Array of coefficients to test for peak constraint.

    Returns
    -------
    v : array
        Vector with 1 where constraint is violated, 0 elsewhere.

    """

    idx = find_peaks(beta, distance=len(beta))[0][0]
    left = np.diff(beta[:idx+1]) < 0
    right = np.diff(beta[idx:]) > 0
    v = np.array(list(left)+list(right))
    return v.astype(np.int)

def check_multi_peak_constraint(beta):
    """Check whether beta contains 2 peaks and is increasing to the first,
    then decreasing, then again increasing and then again decreasing


    Parameters
    ----------
    beta : array
        Array of coefficients to test for multi_peak constraint.

    Returns
    -------
    v : array
        Vector with 1 where constraint is violated, 0 elsewhere.

    """

    peaks, _ = find_peaks(x=beta, prominence=np.std(beta), distance=int(len(beta)/3))
    middle_spline = int(np.mean(peaks))
    v1 = check_peak_constraint(beta=beta[:middle_spline+1])
    v2 = check_peak_constraint(beta=beta[middle_spline:])
    v = np.array(list(v1)+list(v2))
    return v.astype(np.int)

def check_peak_and_valley_constraint(beta):
    """ Check whether beta contains a peak and a valley.
    
    
    Parameters
    ----------
    beta : array
        Array of coefficients to test for peak_and_valley constraint.

    Returns
    -------
    v : array
        Vector with 1 where constraint is violated, 0 elsewhere.

    """

    peak, _ = find_peaks(x=beta, distance=len(beta))
    valley, _ = find_peaks(x=-beta, distance=len(beta))
    middle_spline = int(np.mean([peak, valley]))

    if peak > valley:
        v1 = check_valley_constraint(beta=beta[:middle_spline+1])
        v2 = check_peak_constraint(beta=beta[middle_spline:])
    elif peak < valley:
        v1 = check_peak_constraint(beta=beta[:middle_spline+1])
        v2 = check_valley_constraint(beta=beta[middle_spline:])
    v = np.array(list(v1)+list(v2))
    return v.astype(np.int)


def check_constraint_full_model(model):
    """Checks if the coefficients in the model violate the given constraints for the whole model.
    
    Parameters
    ----------
    model : StarModel
        Instance of StarModel to test the coefficients against the constraints.

    Returns
    -------
    v : array
        Array of boolean whether the constraint is violated. 1 if violated, 0 else.
    """

    assert (model.coef_ is not None), "Please run Model.fit(X, y) first!"
    v = []

    for i, smooth in enumerate(model.smooths):
        beta = model.coef_[model.coef_list[i]:model.coef_list[i+1]]
        constraint = smooth.constraint
        V = check_constraint(beta, constraint=constraint, smooth_type=type(smooth))
        v += list(np.diag(V))
    
    return np.array(v, dtype=np.int)    
    
def bar_chart_of_coefficient_difference_dataframe(df):
    """Takes the dataframe Model.df and plots a bar chart of the rows. 
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of the cofficients values during the iteration of StarModel.fit().
        
    """

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
    """Takes the dataframe Model.df and plots a line chart of the rows. 

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of the cofficients values during the iteration of StarModel.fit().
        
    """

    fig = go.Figure()
    x = np.arange(df.shape[1])
    for i in range(df.shape[0]):
        fig.add_trace(go.Scatter(x=x, y=df.iloc[i], name=f"Iteration {i}",
                                mode="lines"))
    fig.update_layout(title="Coefficients at different iterations",)
    fig.show()

def add_vert_line(fig, x0=0, y0=0, y1=1):
    """Plots a vertical line to the given figure at position x.
    
    Parameters
    ----------
    fig : Plotly.graph_objs
        Figure to plot the vertical line in. 
    x0 : float
        Position of the vertical line.
    y0 : float
        Lowest point of the vertical line.
    y1 : float
        Highest point of the vertical line.
    
    """
    fig.add_shape(dict(type="line", x0=x0, x1=x0, y0=y0, y1=1.2*y1, 
                       line=dict(color="LightSeaGreen", width=1)))
    
def test_model_against_constraint(model, dim=None, plot_=False):
    """Test the model against the constraint on a fine prediction grid. 
    
    Parameters
    ----------
    model : StarModel()
        Instance of StarModel.
    plot_ : boolean
        Indicator whether to plot the violated constraints.
    dim : int
        Number of dimension.

    Returns
    -------
    test : array
        Array of 1s where the constraint is violated, 0 elsewhere.
        
    """
    
    n_samples = 1000
    X_test = np.linspace(0,1,n_samples*dim).reshape((-1, dim))
    y_pred = model.predict(X=X_test)
    v = []
    constraints = [s.constraint for s in model.smooths]
    for constraint in constraints:
        if constraint == "inc":
            test = np.diff(y_pred) < 0
        elif constraint == "dec":
            test = np.diff(y_pred) > 0
        elif constraint == "conv":
            test = np.diff(np.diff(y_pred)) < 0
        elif constraint == "conc":
            test = np.diff(np.diff(y_pred)) > 0
        elif constraint == "peak":
            test = check_peak_constraint(beta=y_pred)
        elif constraint == "multi-peak":
            test = check_multi_peak_constraint(beta=y_pred)
        elif constraint == "valley":
            test = check_valley_constraint(beta=y_pred)
        elif constraint == "multi-valley":
            test = check_multi_valley_constraint(beta=y_pred)
        elif constraint == "peak-and-valley":
            test = check_peak_and_valley_constraint(beta=y_pred)
        v.append(test)

    alltests = [item for sublist in v for item in sublist]
    alltests = np.array(alltests).astype(np.int)
    
    if plot_:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)        
        fig.add_trace(go.Scatter(x=X_test[:,0], y=y_pred, name="Fit"), row=1, col=1)
        fig.add_trace(go.Scatter(x=X_test[:,0], y=test, mode="markers", name=constraint), row=2, col=1)
        fig.show()
    return alltests

