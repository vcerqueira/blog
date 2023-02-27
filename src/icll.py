from typing import List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from collections import Counter


class ICLL:
    """
    Imbalanced Classification via Layered Learning
    """

    def __init__(self, model_l1, model_l2):
        """
        :param model_l1: Predictive model for the first layer
        :param model_l2: Predictive model for the second layer
        """
        self.model_l1 = model_l1
        self.model_l2 = model_l2
        self.clusters = []
        self.mixed_arr = np.array([])

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        :param X: Explanatory variables
        :param y: binary target variable
        """
        assert isinstance(X, pd.DataFrame)
        X = X.reset_index(drop=True)

        if isinstance(y, pd.Series):
            y = y.values

        self.clusters = self.clustering(X=X)

        self.mixed_arr = self.cluster_to_layers(clusters=self.clusters, y=y)

        y_l1 = y.copy()
        y_l1[self.mixed_arr] = 1

        X_l2 = X.loc[self.mixed_arr, :]
        y_l2 = y[self.mixed_arr]

        self.model_l1.fit(X, y_l1)
        self.model_l2.fit(X_l2, y_l2)

    def predict(self, X):
        """
        Predicting new instances
        """

        yh_l1, yh_l2 = self.model_l1.predict(X), self.model_l2.predict(X)

        yh_f = np.asarray([x1 * x2 for x1, x2 in zip(yh_l1, yh_l2)])

        return yh_f

    def predict_proba(self, X):
        """
        Probabilistic predictions
        """

        yh_l1_p = self.model_l1.predict_proba(X)
        try:
            yh_l1_p = np.array([x[1] for x in yh_l1_p])
        except IndexError:
            yh_l1_p = yh_l1_p.flatten()

        yh_l2_p = self.model_l2.predict_proba(X)
        yh_l2_p = np.array([x[1] for x in yh_l2_p])

        yh_fp = np.asarray([x1 * x2 for x1, x2 in zip(yh_l1_p, yh_l2_p)])

        return yh_fp

    @classmethod
    def cluster_to_layers(cls, clusters: List[np.ndarray], y: np.ndarray) -> np.ndarray:
        """
        Defining the layers from clusters
        """

        maj_cls, min_cls, both_cls = [], [], []
        for clst in clusters:
            y_clt = y[np.asarray(clst)]

            if len(Counter(y_clt)) == 1:
                if y_clt[0] == 0:
                    maj_cls.append(clst)
                else:
                    min_cls.append(clst)
            else:
                both_cls.append(clst)

        both_cls_ind = np.array(sorted(np.concatenate(both_cls).ravel()))
        both_cls_ind = np.unique(both_cls_ind)

        if len(min_cls) > 0:
            min_cls_ind = np.array(sorted(np.concatenate(min_cls).ravel()))
        else:
            min_cls_ind = np.array([])

        both_cls_ind = np.unique(np.concatenate([both_cls_ind, min_cls_ind])).astype(int)

        return both_cls_ind

    @classmethod
    def clustering(cls, X, method='ward'):
        """
        Hierarchical clustering analysis
        """

        d = pdist(X)

        Z = linkage(d, method)
        Z[:, 2] = np.log(1 + Z[:, 2])
        sZ = np.std(Z[:, 2])
        mZ = np.mean(Z[:, 2])

        clust_labs = fcluster(Z, mZ + sZ, criterion='distance')

        clusters = []
        for lab in np.unique(clust_labs):
            clusters.append(np.where(clust_labs == lab)[0])

        return clusters
