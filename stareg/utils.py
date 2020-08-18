#!/usr/bin/env python
# coding: utf-8

import copy
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

def check_constraint(beta, constraint):
    """Checks if array beta fits the constraint.
    
    Parameters
    ----------
    beta  : array
        Array of coefficients to be tested against the constraint.
    constraint : str
        Name of the constraint.

    Returns
    -------
    V : np.ndarray
        Matrix with 1 where the constraint is violated, 0 else.
    
    """

    b_diff = np.diff(beta)
    b_diff_diff = np.diff(b_diff)
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
    return np.diag(v)

def check_valley_constraint(beta):
    """Calculate the weight vector v for valley constraint.

    Parameters
    ----------
    beta : array
        Array of coefficients to test for valley constraint.

    Returns
    -------
    v : array
        Vector of with 1 where constraint is violated, 0 elsewhere.

    """

    idx = find_peaks(-beta, distance=len(beta))[0][0]
    left = np.diff(beta[:idx+1]) > 0
    right = np.diff(beta[idx:]) < 0
    v = np.array(list(left)+list(right))
    return v.astype(np.int)

def check_multi_valley_constraint(beta):
    """Check whether beta contains 2 peaks and is increasing to the first,
    then decreasing, then again increasing and then again decreasing
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
        Vector of with 1 where constraint is violated, 0 elsewhere.

    """

    idx = find_peaks(beta, distance=len(beta))[0][0]
    left = np.diff(beta[:idx+1]) < 0
    right = np.diff(beta[idx:]) > 0
    v = np.array(list(left)+list(right))
    return v.astype(np.int)

def check_multi_peak_constraint(beta):
    """Check whether beta contains 2 peaks and is increasing to the first,
    then decreasing, then again increasing and then again decreasing
    """

    peaks, _ = find_peaks(x=beta, prominence=np.std(beta), distance=int(len(beta)/3))
    middle_spline = int(np.mean(peaks))
    v1 = check_peak_constraint(beta=beta[:middle_spline+1])
    v2 = check_peak_constraint(beta=beta[middle_spline:])
    v = np.array(list(v1)+list(v2))
    return v.astype(np.int)

def check_peak_and_valley_constraint(beta):
    """ Check whether beta contains a peak and a valley """
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
        V = check_constraint(beta, constraint=constraint)
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
    
def test_model_against_constraint(model, plot_=False):
    """Test the model against the constraint. 
    
    If the model fails the constraint, place a 1 at the point, else place a 0. 
    """
        
    x_test = np.linspace(0,1,10000)
    bs = copy.deepcopy(model.smooths[0])
    bs.bspline_basis(x_data=x_test, k=model.smooths[0].n_param)
    y_pred = bs.basis @ model.coef_
    constraint = model.smooths[0].constraint
    
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
    
    test = test.astype(np.int)
    
    if plot_:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)        
        fig.add_trace(go.Scatter(x=x_test, y=y_pred, name="Fit"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_test, y=test, mode="markers", name=constraint), row=2, col=1)
        fig.show()
    return test

