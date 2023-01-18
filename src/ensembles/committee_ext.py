import pandas as pd

from src.utils.proportion import proportion


class ExternalCommittee:

    def __init__(self, omega, weights, col_names, n_models=None):
        if n_models is not None:
            self.n_models = n_models
        else:
            self.n_models = int(omega * weights.shape[1])

            if self.n_models < 1:
                self.n_models = 1

        self.col_names = col_names
        self.n_bad_models = weights.shape[1] - self.n_models

    def from_weights(self,
                     target_weights: pd.DataFrame,
                     source_weights: pd.DataFrame):

        assert target_weights.shape == source_weights.shape

        if not isinstance(target_weights, pd.DataFrame):
            target_weights = pd.DataFrame(target_weights, columns=self.col_names)

        if not isinstance(source_weights, pd.DataFrame):
            source_weights = pd.DataFrame(source_weights, columns=self.col_names)

        weights = target_weights.copy()

        for i in range(weights.shape[0]):
            w = source_weights.iloc[i, :].copy()

            zero_value = w.sort_values()[:self.n_bad_models].index.to_list()

            weights.iloc[i, :][zero_value] = 0

            weights.iloc[i, :] = proportion(weights.iloc[i, :])

        out = weights.values

        return out
