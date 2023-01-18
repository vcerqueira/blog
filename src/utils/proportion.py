import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def min_max_norm_vector(x: pd.Series):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    scaler = MinMaxScaler()
    w0 = scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
    w0 = pd.Series(w0, index=x.index)

    return w0


def neg_normalize_and_proportion(x):
    """ Min max normalization followed by proportion
    """
    return proportion(min_max_norm_vector(-x))


def proportion(x):
    """ Proportion of sum
    """
    return x / np.sum(x)


def normalize_and_proportion(x):
    """ Min max normalization followed by proportion
    """
    return proportion(min_max_norm_vector(x))
