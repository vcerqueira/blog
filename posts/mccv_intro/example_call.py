from sklearn.datasets import make_regression

from src.mccv import MonteCarloCV

X, y = make_regression(n_samples=120)

mccv = MonteCarloCV(n_splits=5,
                    train_size=0.6,
                    test_size=0.1,
                    gap=0)

for train_index, test_index in mccv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
