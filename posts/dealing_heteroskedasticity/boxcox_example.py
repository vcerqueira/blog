import numpy as np
from pmdarima.datasets import load_airpassengers
from scipy.stats import boxcox
from scipy.special import inv_boxcox

# loading the data
series = load_airpassengers(True)

# transforming the series
# lambda_ is the transformation parameter
series_transformed, lambda_ = boxcox(series)

# reverting to the original scale
original_series = inv_boxcox(series_transformed, lambda_)

# check if it is the same as the original data
np.allclose(original_series, series)
# True
