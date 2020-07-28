#!/usr/bin/env python
# coding: utf-8

import numpy as np
import plotly.graph_objs as go
from scipy.signal import find_peaks


def check_constraint(beta, constraint):
    """Checks if beta fits the constraint.
    
    Parameters:
    ---------------
    beta  : array     - Array of coefficients to be tested against the constraint.
    constraint : str  - Name of the constraint.
    
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
    elif constraint == "smooth":
        v = list(np.ones(len(b_diff_diff), dtype=np.int)) #+ [0,0]
    elif constraint == "peak":
        v = check_peak_constraint(beta=beta)
    elif constraint == "valley":
        v = check_valley_constraint(beta=beta)
    return np.diag(v)

def check_valley_constraint(beta):
    """Calculate the weight vector v for violated constraint.

    Parameters:
    ------------
    beta   : array              - array of coefficients to test for valley

    Returns:
    ------------
    v      : array              - vector of with 1 where constraint is violated,
                                  0 elsewhere
    """
    idx = find_peaks(-beta, distance=len(beta))[0][0]
    left = list(np.diff(beta[:idx]) > 0)
    right = list(np.diff(beta[idx:]) < 0)
    v = np.array(left+right+[False])
    return v.astype(np.int)


def check_peak_constraint(beta):
    """Calculate the weight vector v for violated constraint.

    Parameters:
    ------------
    beta   : array              - array of coefficients to test for peak

    Returns:
    ------------
    v      : array              - vector of with 1 where constraint is violated,
                                  0 elsewhere
    """
    idx = find_peaks(beta, distance=len(beta))[0][0]
    left = list(np.diff(beta[:idx]) < 0)
    right = list(np.diff(beta[idx:]) > 0)
    v = np.array(left+right+[False])
    return v.astype(np.int)

def check_constraint_full_model(model, y):
    """Checks if the coefficients in the model violate the given constraints.
    
    Parameters:
    -------------
    model : class StarModel()       - instance of StarModel to test the constraints for
    y     : array                   - target data for the peak/valley test

    Returns:
    -------------
    v : list   - list of boolean wheter the constraint is violated. 
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
    """Takes the dataframe Model.df and plots a bar chart of the rows. """

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
    """Takes the dataframe Model.df and plots a line chart of the rows. """

    fig = go.Figure()
    x = np.arange(df.shape[1])

    for i in range(df.shape[0]):
        fig.add_trace(go.Scatter(x=x, y=df.iloc[i], name=f"Iteration {i}",
                                mode="lines"))

    fig.update_layout(title="Coefficients at different iterations",)
    fig.show()

def add_vert_line(fig, x0=0, y0=0, y1=1):
    """ plots a vertical line to the given figure at position x"""
    fig.add_shape(dict(type="line", x0=x0, x1=x0, y0=y0, y1=1.2*y1, 
                       line=dict(color="LightSeaGreen", width=1)))
    


def find_peak_and_basis(x=None, y=None):
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy.signal import find_peaks
    from Smooth import Smooths
    
    np.random.seed(42)

    if x is None and y is None:
        x = np.linspace(-3,3,1000)
        y = np.sin(x) + 0.1*np.random.randn(len(x))
    
    peaks, properties = find_peaks(y, distance=int(len(x)))
    x_peak, y_peak = x[peaks], y[peaks]

    s = Smooths(x_data=x, n_param=20)

    fig = go.Figure()
    for i in range(s.basis.shape[1]):
        fig.add_trace(go.Scatter(x=s.x, y=s.basis[:,i], 
                                 name=f"BSpline {i+1}", mode="lines"))
    for i in s.knots:
        add_vert_line(fig, x0=i)

    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data"))
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=x_peak, 
            y=y_peak,
            marker=dict(color="red", size=18, line=dict(color="DarkSlateGrey", width=2)),
            name="Peak Locations"
        )

    )   
    fig.show()
