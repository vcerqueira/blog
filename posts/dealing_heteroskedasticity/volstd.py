import re
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae

from src.tde import time_delay_embedding

# https://github.com/vcerqueira/blog/tree/main/data
wine = pd.read_csv('data/wine_sales.csv', parse_dates=['date'])
wine.set_index('date', inplace=True)

# train test split
train, test = train_test_split(wine, test_size=0.2, shuffle=False)

# transforming the time series for supervised learning
train_df, test_df = [], []
for col in wine:
    # using 12 lags to forecast the next value (horizon=1)
    col_train_df = time_delay_embedding(train[col], n_lags=12, horizon=1)
    col_train_df = col_train_df.rename(columns=lambda x: re.sub(col, 'Series', x))
    train_df.append(col_train_df)

    col_test_df = time_delay_embedding(test[col], n_lags=12, horizon=1)
    col_test_df = col_test_df.rename(columns=lambda x: re.sub(col, 'Series', x))
    test_df.append(col_test_df)

# different series are concatenated on rows
# to train a global forecasting model
train_df = pd.concat(train_df, axis=0)
test_df = pd.concat(test_df, axis=0)

# splitting the explanatory variables from target variables
predictor_variables = train_df.columns.str.contains('\(t\-')
target_variables = train_df.columns.str.contains('Series\(t\+')
X_train = train_df.iloc[:, predictor_variables]
Y_train = train_df.iloc[:, target_variables]
X_test = test_df.iloc[:, predictor_variables]
Y_test = test_df.iloc[:, target_variables]

# volatility standardization
X_train_vs = X_train.apply(lambda x: x / x.std(), axis=0)
X_test_vs = X_test.apply(lambda x: x / x.std(), axis=0)

mod_raw = XGBRegressor()
mod_vs = XGBRegressor()
mod_log = XGBRegressor()

# fitting on raw data
mod_raw.fit(X_train, Y_train)
# fitting with log-scaled data
mod_log.fit(np.log(X_train), np.log(Y_train))
# fitting with vol. std. data
mod_vs.fit(X_train_vs, Y_train)

# making predictions
preds_raw = mod_raw.predict(X_test)
preds_log = np.exp(mod_log.predict(np.log(X_test)))
preds_vs = mod_vs.predict(X_test_vs)

print(mae(Y_test, preds_raw))
# 301.73
print(mae(Y_test, preds_vs))
# 294.74
print(mae(Y_test, preds_log))
# 308.41
