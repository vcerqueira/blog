from plotnine import *
from sklearn.datasets import make_regression

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
from src.cv_extensions.blocked_cv import BlockedKFold
from src.cv_extensions.hv_blocked_cv import hvBlockedKFold
from src.cv_extensions.modified_cv import ModifiedKFold
from src.cv_extensions.sliding_tss import SlidingTimeSeriesSplit
from src.cv_extensions.holdout import Holdout
from src.mccv import MonteCarloCV

from src.cv_plot import cv_plot_points

N_SPLITS = 5
N = 30
GAP = 3
WIDTH, HEIGHT = 9, 5

X, y = make_regression(n_samples=N)

kfcv = KFold(n_splits=N_SPLITS, shuffle=True)
stss_cv = SlidingTimeSeriesSplit(n_splits=N_SPLITS)
tss_cv = TimeSeriesSplit(n_splits=N_SPLITS)
tssg_cv = TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP)
bcv = BlockedKFold(n_splits=N_SPLITS)
hvbcv = hvBlockedKFold(n_splits=N_SPLITS, gap=GAP)
mcv = ModifiedKFold(n_splits=N_SPLITS, gap=2)
mccv = MonteCarloCV(n_splits=N_SPLITS, train_size=.5, test_size=.1)
ho = Holdout(n=N, test_size=.3)

plot_kfcv, _ = cv_plot_points(kfcv, X, y)
plot_stss_cv, _ = cv_plot_points(stss_cv, X, y)
plot_tss_cv, _ = cv_plot_points(tss_cv, X, y)
plot_tssg_cv, _ = cv_plot_points(tssg_cv, X, y)
plot_bcv, _ = cv_plot_points(bcv, X, y)
plot_hvbcv, _ = cv_plot_points(hvbcv, X, y)
plot_mcv, _ = cv_plot_points(mcv, X, y)
plot_mccv, _ = cv_plot_points(mccv, X, y)
plot_ho, _ = cv_plot_points(ho, X, y)
plot_ho += ylim(1, 1)

plot_kfcv.save('kfcv.pdf', width=WIDTH, height=HEIGHT)
plot_stss_cv.save('stss.pdf', width=WIDTH, height=HEIGHT)
plot_tss_cv.save('tss.pdf', width=WIDTH, height=HEIGHT)
plot_tssg_cv.save('tssg.pdf', width=WIDTH, height=HEIGHT)
plot_bcv.save('bcv.pdf', width=WIDTH, height=HEIGHT)
plot_hvbcv.save('hvbcv.pdf', width=WIDTH, height=HEIGHT)
plot_mcv.save('mcv.pdf', width=WIDTH, height=HEIGHT)
plot_mccv.save('mccv.pdf', width=WIDTH, height=HEIGHT)
plot_ho.save('ho.pdf', width=WIDTH, height=HEIGHT)
