import datetime
import numpy as np
import pandas as pd
from joblib import dump, load
from plotnine import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import (RandomizedSearchCV,
                                     TimeSeriesSplit,
                                     train_test_split)

from src.tde import time_delay_embedding
from src.plots.ts_lineplot_simple import LinePlot

START_DATE = '2022-01-01'
N_LAGS, HORIZON = 24, 24
URL = f'https://erddap.marine.ie/erddap/tabledap/IWaveBNetwork.csv?time%2CSignificantWaveHeight&time%3E={START_DATE}T00%3A00%3A00Z&station_id=%22AMETS%20Berth%20B%20Wave%20Buoy%22'


def reading_data(url: str) -> pd.Series:
    """
    Reading ERDAP data

    :param url: ERDAP url as string
    :return: hourly wave height time series as pd.Series
    """

    # reading data directly from erdap
    data = pd.read_csv(url, skiprows=[1], parse_dates=['time'])

    # setting time to index and getting the target series
    series = data.set_index('time')['SignificantWaveHeight']

    # transforming data to hourly and from centimeters to meters
    series_hourly = series.resample('H').mean() / 100

    return series_hourly


class LogTransformation:

    @staticmethod
    def transform(x):
        xt = np.sign(x) * np.log(np.abs(x) + 1)

        return xt

    @staticmethod
    def inverse_transform(xt):
        x = np.sign(xt) * (np.exp(np.abs(xt)) - 1)

        return x


def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """
    param X: lagged observations (explanatory variables)

    :return: new features
    """

    summary_stats = {'mean': np.mean, 'sdev': np.std}

    features = {}
    for f in summary_stats:
        features[f] = X.apply(lambda x: summary_stats[f](x), axis=1)

    features_df = pd.concat(features, axis=1)
    X_feats = pd.concat([X, features_df], axis=1)

    return X_feats


series = reading_data(URL)

plot = LinePlot.univariate(series.reset_index(),
                           x_axis_col='time',
                           y_axis_col='SignificantWaveHeight',
                           line_color='#0058ab', y_lab='Wave Height (m)')

train, test = train_test_split(series, test_size=0.2, shuffle=False)

X_train, Y_train = time_delay_embedding(train, n_lags=N_LAGS, horizon=HORIZON, return_Xy=True)
X_test, Y_test = time_delay_embedding(test, n_lags=N_LAGS, horizon=HORIZON, return_Xy=True)

X_train = LogTransformation.transform(X_train)
Y_train = LogTransformation.transform(Y_train)

X_train_ext = feature_engineering(X_train)

# time series cv procedure
tscv = TimeSeriesSplit(n_splits=5, gap=50)

# defining the search space
# a simple optimization of the number of trees of a RF
model = RandomForestRegressor()
param_search = {'n_estimators': [10, 50, 100, 300],
                'criterion': ['squared_error', 'absolute_error'],
                'max_depth': [None, 2, 5, 10],
                'max_features': ['log2', 'sqrt']}

# applying CV with a gridsearch on the training data
gs = RandomizedSearchCV(estimator=model,
                        cv=tscv,
                        refit=True,
                        param_distributions=param_search,
                        n_iter=10, n_jobs=1)

gs.fit(X_train_ext, Y_train)

# inference on test set and evaluation
X_test = LogTransformation.transform(X_test)
X_test_ext = feature_engineering(X_test)
preds_log = gs.predict(X_test_ext)

# reverting the log transformation
preds = LogTransformation.inverse_transform(preds_log)

estimated_performance = r2_score(Y_test, preds)

# preparing all available data for auto-regression
X, Y = time_delay_embedding(series, n_lags=N_LAGS, horizon=HORIZON, return_Xy=True)

# applying preprocessing steps
X = LogTransformation.transform(X)
Y = LogTransformation.transform(Y)
X_ext = feature_engineering(X)

# model fitting
final_model = RandomForestRegressor(**gs.best_params_)
final_model.fit(X_ext, Y)

dump(final_model, 'random_forest_v1.joblib')

## forecasting

yesterday = datetime.date.today() - datetime.timedelta(days=1)
yesterday = yesterday.strftime('%Y-%m-%d')

LIVE_URL = f'https://erddap.marine.ie/erddap/tabledap/IWaveBNetwork.csv?time%2CSignificantWaveHeight&time%3E={yesterday}T00%3A00%3A00Z&station_id=%22AMETS%20Berth%20B%20Wave%20Buoy%22'

new_series = reading_data(LIVE_URL)

lags = new_series.tail(N_LAGS)
lags_df = pd.DataFrame(lags).T.reset_index(drop=True)
lags_df.columns = X.columns

lags_df = LogTransformation.transform(lags_df)
lags_feats = feature_engineering(lags_df)

final_model = load('random_forest_v1.joblib')

log_forecasts = final_model.predict(lags_feats)

# reverting the log transformation
forecasts = LogTransformation.inverse_transform(log_forecasts)


## Plotting

forecasts_all = [tree.predict(lags_feats)
                 for tree in final_model.estimators_]
forecasts_df = pd.DataFrame(np.asarray(forecasts_all).reshape(100, 24)).T
forecasts_df.index = pd.date_range(start=new_series.index[-1], periods=HORIZON + 1, freq='H')[1:]

forecasts_df = LogTransformation.inverse_transform(forecasts_df)

forecasts_df.index.name = 'Time'
lags.index.name = 'Time'

yhat = forecasts_df.reset_index().melt('Time')

lagsr = lags.reset_index()
lagsr.columns = ['Time', 'value']
lagsr['variable'] = 'Lags'

df = pd.concat([lagsr, yhat], axis=0)
df['variable'] = pd.Categorical(df['variable'], categories=df['variable'].unique())
df['Type'] = 'Ocean wave height forecasts for the next 24 hours'

avg_yhat = yhat.groupby('Time').mean().reset_index()

plot = \
    ggplot(df) + \
    aes(x='Time',
        color='variable',
        y='value') + \
    theme_538(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.2,
          axis_text=element_text(size=12),
          axis_text_x=element_text(size=10),
          legend_title=element_blank(),
          legend_position='none')

plot += geom_line(size=1, alpha=.3)
plot += geom_line(data=avg_yhat,
                  mapping=aes(x='Time',
                              y='value',
                              group=1),
                  size=3, color='#0d3aa9')
plot += geom_line(data=lagsr,
                  mapping=aes(x='Time',
                              y='value',
                              group=1),
                  size=1.3, color='#cf8806')
plot += facet_wrap('~ Type', nrow=1)

plot = \
    plot + \
    xlab('') + \
    ylab('Wave Height (m)') + \
    ggtitle('')

print(plot)

plot.save('forecasts_live.pdf', width=10, height=7)
