import numpy as np
from scipy import stats
import statsmodels.tsa.stattools as ts


def covariance(x: np.ndarray, y: np.ndarray) -> float:
    """ Covariance between x and y
    """
    cov_xy = np.cov(x, y)[0][1]

    return cov_xy


def co_integration(x: np.ndarray, y: np.ndarray):
    """ Co-integration test between x and y
    """
    r, _, _ = ts.coint(x, y)

    return r


def correlation(x: np.ndarray,
                y: np.ndarray,
                method: str = "kendall"):
    """ Correlation between x and y
    """
    assert method in ["pearson", "spearman", "kendall"]

    corr, p_value = stats.kendalltau(x, y)

    return corr
