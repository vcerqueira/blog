from typing import Union, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.ensembles.committee_ext import ExternalCommittee
from src.utils.proportion import neg_normalize_and_proportion


class Arbitrating:
    """
    ADE
    """

    def __init__(self,
                 lambda_: int = 50,
                 committee_size: Optional[float] = None):
        """
        :param lambda_: No of recent observations used to trim ensemble
        """
        self.lambda_ = lambda_
        self.meta_model = RandomForestRegressor()
        self.committee_size = committee_size
        self.col_names = None

    def fit(self,
            Y_hat_insample: pd.DataFrame,
            y_insample: Union[pd.Series, np.ndarray],
            X_tr: pd.DataFrame):

        self.col_names = Y_hat_insample.columns

        Y_hat_insample.reset_index(drop=True, inplace=True)

        if isinstance(y_insample, pd.Series):
            y_insample = y_insample.values

        base_loss = Y_hat_insample.apply(func=lambda x: x - y_insample, axis=0)
        base_loss.reset_index(drop=True, inplace=True)

        self.meta_model.fit(X_tr.reset_index(drop=True), base_loss)

    def get_weights(self,
                    X: pd.DataFrame,
                    source_weights: Optional[pd.DataFrame] = None):

        E_hat = self.meta_model.predict(X)
        E_hat = pd.DataFrame(E_hat).abs()
        E_hat.columns = self.col_names

        W = E_hat.apply(
            func=lambda x: neg_normalize_and_proportion(x),
            axis=1)

        if source_weights is None:
            return W
        else:
            if self.committee_size is None:
                self.committee_size = 1

            # n_models = int(self.committee_size * E_hat.shape[1])

            committee = ExternalCommittee(omega=self.committee_size,
                                          col_names=E_hat.columns,
                                          weights=source_weights)

            W = committee.from_weights(target_weights=W,
                                       source_weights=source_weights)

            W = pd.DataFrame(W, columns=self.col_names)

            return W
