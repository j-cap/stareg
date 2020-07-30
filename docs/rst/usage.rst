
Usage
=====


1-D case
--------

The a stareg model is fit to an Gaussian function overlayed with uniform noise:

::

    import stareg
    from stareg.star_model import StarModel 

    import numpy as np 

    # create some data
    n_samples, n_param = 100, 25
    noise = np.random.random(100)*0.2
    x = np.linspace(0, 1, n_samples)
    y = np.exp(-(x - 0.4)**2 / 0.01) + noise

    # reshape to suitable dimensions
    x, y = x.reshape(-1, 1), y.ravel()

    # create a description tuple for StarModel
    description = ( ("s(1)", "peak", n_param, (0.1, 100), "equidistant"), )

    # create the model and fit it to the data
    Model = StarModel(description=description)
    Model.fit(X=x, y=y, plot_=True)


2-D case
--------

This stareg model is fit to the famous `monkey saddle <https://en.wikipedia.org/wiki/Monkey_saddle>`_ function:

::

    import stareg
    from stareg.star_model import StarModel

    import numpy as np

    # create some data
    n_samples, n_param = 100, 25
    x1, x2 = np.linspace(-1, 1, n_samples), np.linspace(-3,3, n_samples)
    noise = np.random.random(n_samples) * 2
    y = x1**3 - 3*x1*x2**2 + noise

    # reshape to suitable dimensions
    x = np.array([x1, x2]).T

    # create a description tuple for StarModel
    description = ( ("s(1)", "smooth", n_param, (0.1, 100), "equidistant"), 
                    ("s(2)", "dec", n_param, (0.1, 100), "quantile"), )

    # create the model and fit it to the data
    Model = StarModel(description=description)
    Model.fit(X=x, y=y, plot_=True) 


Multi-D case
------------

In general, arbitrary dimensions are possible. Just include more tuples in the description tuple. 