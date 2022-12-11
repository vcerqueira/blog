from sklearn.model_selection import TimeSeriesSplit
from sklearn.datasets import make_regression

from src.cv_plot import cv_plot
from src.mccv import MonteCarloCV

N_SPLITS = 5

X, y = make_regression(n_samples=120)

mccv = MonteCarloCV(n_splits=N_SPLITS, train_size=0.6, test_size=0.1, gap=0)
mccv_gap = MonteCarloCV(n_splits=N_SPLITS, train_size=0.6, test_size=0.1, gap=10)
tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=0)
tscv_gap = TimeSeriesSplit(n_splits=N_SPLITS, gap=10)

plot_mccv_cont = cv_plot(mccv, X, y)
plot_mccv_gap = cv_plot(mccv_gap, X, y)
plot_tscv_cont = cv_plot(tscv, X, y)
plot_tscv_gap = cv_plot(tscv_gap, X, y)

plot_mccv_cont.save('mccv.pdf', height=4, width=7)
plot_tscv_cont.save('tscv.pdf', height=4, width=7)
plot_mccv_gap.save('mccv_gap.pdf', height=4, width=7)
