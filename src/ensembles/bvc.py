import numpy as np
import pandas as pd


class BiasVarianceCovariance:

    @classmethod
    def get_bvc(cls, y_hat: pd.DataFrame, y: np.ndarray):
        return cls.avg_sqr_bias(y_hat, y), cls.avg_var(y_hat), cls.avg_cov(y_hat)

    @staticmethod
    def avg_sqr_bias(y_hat: pd.DataFrame, y: np.ndarray):
        """
        :param y_hat: predictions as pd.DataFrame with shape (n_observations, n_models).
        The predictions of each model are in different columns
        :param y: actual values as np.array
        """
        return ((y_hat.mean(axis=0) - y.mean()).mean()) ** 2

    @staticmethod
    def avg_var(y_hat: pd.DataFrame):
        M = y_hat.shape[1]

        return y_hat.var().mean() / M

    @staticmethod
    def avg_cov(y_hat: pd.DataFrame):
        M = y_hat.shape[1]

        cov_df = pd.DataFrame(np.cov(y_hat))
        np.fill_diagonal(cov_df.values, 0)

        cov_term = cov_df.values.sum() * (1 / (M * (M - 1)))

        return cov_term
