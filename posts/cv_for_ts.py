from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from src.cv_plot import cv_plot

X, y = make_regression(n_samples=120)

kfcv = KFold(n_splits=5, shuffle=False)
tscv = TimeSeriesSplit(n_splits=5, gap=0)
tscv_gap = TimeSeriesSplit(n_splits=5, gap=5)

plot_cont = cv_plot(tscv, X, y)
plot_gap = cv_plot(tscv_gap, X, y)
plot_kfcv = cv_plot(kfcv, X, y)

model = RandomForestRegressor()
param_search = {'n_estimators': [10, 100]}

gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search)
gsearch.fit(X, y)
