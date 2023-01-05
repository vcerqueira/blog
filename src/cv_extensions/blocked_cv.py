from sklearn.model_selection import KFold


class BlockedKFold(KFold):

    def __init__(self,
                 n_splits: int = 5,
                 random_state: int = None):
        super().__init__(n_splits=n_splits,
                         shuffle=False,
                         random_state=random_state)
