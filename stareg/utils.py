#!/usr/bin/env python
# coding: utf-8

import numpy as np
import plotly.graph_objs as go
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
        v = [0 if i > 0 else 1 for i in b_diff] #+ [0]
    elif constraint == "dec":
        v = [0 if i < 0 else 1 for i in b_diff] #+ [0]
    elif constraint == "conv":
        v = [0 if i > 0 else 1 for i in b_diff_diff] #+ [0,0]
    elif constraint == "conc":
        v = [0 if i < 0 else 1 for i in b_diff_diff] #+ [0,0]
    elif constraint == "smooth":
        v = list(np.ones(len(b_diff_diff), dtype=np.int)) #+ [0,0]
    elif constraint == "peak":
        v = check_peak_constraint(beta=beta)
    elif constraint == "multi-peak":
        v = check_multi_peak_constraint(beta=beta)
    elif constraint == "valley":
        v = check_valley_constraint(beta=beta)
    elif constraint == "multi-valley":
        v = check_multi_valley_constraint(beta=beta)
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
    left = list(np.diff(beta[:idx]) > 0)
    right = list(np.diff(beta[idx:]) < 0)
    v = np.array(left+right+[False])
    return v.astype(np.int)

def check_multi_valley_constraint(beta):
    """Check whether beta contains 2 peaks and is increasing to the first,
    then decreasing, then again increasing and then again decreasing
    """

    peaks, properties = find_peaks(x=-beta, prominence=beta.max()/5, distance=int(len(beta)/10))
    middle_spline = int(np.mean(peaks))
    v1 = check_valley_constraint(beta=beta[:middle_spline])
    v2 = check_valley_constraint(beta=beta[middle_spline:])
    v = np.array(list(v1)+list(v2)+[False])
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
    left = list(np.diff(beta[:idx]) < 0)
    right = list(np.diff(beta[idx:]) > 0)
    v = np.array(left+right+[False])
    return v.astype(np.int)

def check_multi_peak_constraint(beta):
    """Check whether beta contains 2 peaks and is increasing to the first,
    then decreasing, then again increasing and then again decreasing
    """

    peaks, properties = find_peaks(x=beta, prominence=beta.max()/5, distance=int(len(beta)/10))
    middle_spline = int(np.mean(peaks))
    v1 = check_peak_constraint(beta=beta[:middle_spline])
    v2 = check_peak_constraint(beta=beta[middle_spline:])
    v = np.array(list(v1)+list(v2)+[False])
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
    


