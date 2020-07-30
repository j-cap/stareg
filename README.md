
# stareg - **ST**ructured **A**dditive **REG**ression


> Look how easy it is to use:
> 
> -- <cite>anonymous</cite>

```python
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

# create a description tuple for the StarModel
description = ( ("s(1)", "peak", n_param, (0.1, 100), "equidistant"), )

# create the model and fit it to the data
Model = StarModel(description=description)
Model.fit(X=x, y=y, plot_=True)
```


## Features

* Structured Additive Regression
* B-Splines and P-Splines
* Incorporate prior knowledge using constraints: 
  * Monotonicity constraints: increasing, decreasing
  * Shape constraints: convex, concave
  * Peak and Valley constraints
* Interpretable Machine Learning
* Uni- and multivariate regression

## Tags

* [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning>) 
* [XAI](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence>) 
* [Regression](https://en.wikipedia.org/wiki/Regression_analysis)
* [Splines](https://en.wikipedia.org/wiki/Spline_(mathematics))


## Installation

Install stareg by running:
```
pip install stareg
```


## Contribute
- Issue Tracker: [GitHub Repo](https://github.com/j-cap/stareg/issues)
- Source Code: [GitHub Repo](https://github.com/j-cap/stareg/src/)

## Support
If you are having issues, please let us know. 
Contact can be made via [GitHub](https://github.com/j-cap/stareg)

## License
The project is licensed under the MIT license.