from typing import Union

import numpy as np
import pandas as pd


class Oracle:

    def __init__(self):
        pass

    def fit(self, Y_hat: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        pass

    @staticmethod
    def get_weights(Y_hat: pd.DataFrame, y: pd.Series):
        se = Y_hat.apply(func=lambda x: (x - y) ** 2, axis=0)

        for i, row in se.iterrows():
            max_idx = row.argmin()
            row -= row
            row[max_idx] = 1

        out = se.values

        return out