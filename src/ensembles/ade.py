from typing import Union, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.utils.proportion import neg_normalize_and_proportion


class Arbitrating:
    """
    ADE - arbitrated dynamic ensemble

    A metamodel is trained to forecast the error of individual models
    This error is then normalized to a 0-1 scale
    """

    def __init__(self):
        """

        param committee_size: Ratio of ensembles to keep at each instant after pruning.
        If None (default), all models are used.
        """

        self.meta_model = RandomForestRegressor()
        self.col_names = None

    def fit(self,
            Y_hat_insample: pd.DataFrame,
            y_insample: Union[pd.Series, np.ndarray],
            X_tr: pd.DataFrame):
        """
        Fitting the metamodel using training forecasts

        :param Y_hat_insample: Forecasts of each model in the training data as pd.DF
        :param y_insample: Actual training values for computing residuals
        :param X_tr: Training explanatory variables as pd.DF

        :return: self, with trained metamodel
        """

        self.col_names = Y_hat_insample.columns

        Y_hat_insample.reset_index(drop=True, inplace=True)

        if isinstance(y_insample, pd.Series):
            y_insample = y_insample.values

        base_loss = Y_hat_insample.apply(func=lambda x: x - y_insample, axis=0)
        base_loss.reset_index(drop=True, inplace=True)

        self.meta_model.fit(X_tr.reset_index(drop=True), base_loss)

    def get_weights(self, X: pd.DataFrame):
        """
        Predict the weights of each model for an input data set


        param X: Input explanatory variables (e.g. lagged features)
        param source_weights: External weights for trimming the ensemble. Defaults to None (all models are used)

        return: Weights for each model in the ensemble
        """

        E_hat = self.meta_model.predict(X)
        E_hat = pd.DataFrame(E_hat).abs()
        E_hat.columns = self.col_names

        W = E_hat.apply(
            func=lambda x: neg_normalize_and_proportion(x),
            axis=1)

        return W
