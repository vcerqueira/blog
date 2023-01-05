from abc import ABC

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


class SlidingTimeSeriesSplit(TimeSeriesSplit, ABC):

    def __init__(self, n_splits: int, gap: int = 0):
        super().__init__(n_splits=n_splits, gap=gap)
        self.n = -1

    def split(self, X, y=None, groups=None):
        self.n = X.shape[0]
        test_size = int(self.n // self.n_splits)

        self.max_train_size = test_size

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap = self.gap
        test_size = (
            self.test_size if self.test_size is not None else n_samples // n_folds
        )

        # Make sure we have enough samples for the given split parameters
        if n_folds > n_samples:
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater"
                f" than the number of samples={n_samples}."
            )
        if n_samples - gap - (test_size * n_splits) <= 0:
            raise ValueError(
                f"Too many splits={n_splits} for number of samples"
                f"={n_samples} with test_size={test_size} and gap={gap}."
            )

        indices = np.arange(n_samples)
        test_starts = range(n_samples - n_splits * test_size, n_samples, test_size)

        for test_start in test_starts:
            train_end = test_start - gap
            if self.max_train_size and self.max_train_size < train_end:
                yield (
                    indices[train_end - self.max_train_size: train_end],
                    indices[test_start: test_start + test_size],
                )
            else:
                yield (
                    indices[:train_end],
                    indices[test_start: test_start + test_size],
                )
