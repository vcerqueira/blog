import re

import pandas as pd
import numpy as np


def time_delay_embedding(series: pd.Series,
                         n_lags: int,
                         horizon: int,
                         return_Xy: bool = False):
    """
    Time delay embedding
    Time series for supervised learning

    :param series: time series as pd.Series
    :param n_lags: number of past values to used as explanatory variables
    :param horizon: how many values to forecast
    :param return_Xy: whether to return the lags split from future observations

    :return: pd.DataFrame with reconstructed time series
    """
    print(series)
    assert isinstance(series, pd.Series)

    if series.name is None:
        name = 'Series'
    else:
        name = series.name

    n_lags_iter = list(range(n_lags, -horizon, -1))

    df_list = [series.shift(i) for i in n_lags_iter]
    df = pd.concat(df_list, axis=1).dropna()
    df.columns = [f'{name}(t-{j - 1})'
                  if j > 0 else f'{name}(t+{np.abs(j) + 1})'
                  for j in n_lags_iter]

    df.columns = [re.sub('t-0', 't', x) for x in df.columns]

    if not return_Xy:
        return df

    is_future = df.columns.str.contains('\+')

    X = df.iloc[:, ~is_future]
    Y = df.iloc[:, is_future]
    if Y.shape[1] == 1:
        Y = Y.iloc[:, 0]

    return X, Y


def from_matrix_to_3d(df: pd.DataFrame) -> np.ndarray:
    """
    Transforming a time series from matrix into 3-d structure for deep learning
    :param df: (pd.DataFrame) Time series in the matrix format after embedding

    :return: Reshaped time series into 3-d structure
    """
    cols = df.columns

    var_names = np.unique([re.sub(r'\([^)]*\)', '', c) for c in cols]).tolist()

    arr_by_var = [df.loc[:, cols.str.contains(v)].values for v in var_names]
    arr_by_var = [x.reshape(x.shape[0], x.shape[1], 1) for x in arr_by_var]

    ts_arr = np.concatenate(arr_by_var, axis=2)

    return ts_arr


def from_3d_to_matrix(arr: np.ndarray, col_names: pd.Index):
    if arr.shape[2] > 1:
        arr_split = np.dsplit(arr, arr.shape[2])
    else:
        arr_split = [arr]

    arr_reshaped = [x.reshape(x.shape[0], x.shape[1]) for x in arr_split]

    df = pd.concat([pd.DataFrame(x) for x in arr_reshaped], axis=1)

    df.columns = col_names

    return df
