import numpy as np
from sklearn.model_selection import KFold

from sklearn.utils import indexable, check_random_state
from sklearn.utils.validation import _num_samples


class ModifiedKFold(KFold):

    def __init__(self,
                 n_splits: int,
                 gap: int = 1,
                 random_state: int = None):
        super().__init__(n_splits=n_splits,
                         shuffle=True,
                         random_state=random_state)

        self.gap = gap

    def split(self, X, y=None, groups=None):

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                (
                    "Cannot have number of splits n_splits={0} greater"
                    " than the number of samples: n_samples={1}."
                ).format(self.n_splits, n_samples)
            )

        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]

            if self.gap > 0:
                dependent_obs = [np.arange(x - self.gap, x + self.gap + 1, 1) for x in test_index]
                gap_zone = np.unique(np.concatenate(dependent_obs))

                train_index = np.setdiff1d(train_index, gap_zone)

            yield train_index, test_index

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[: n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop
