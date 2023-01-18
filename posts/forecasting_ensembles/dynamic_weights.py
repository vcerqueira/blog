from plotnine import *
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNetCV

from sklearn.metrics import mean_absolute_error as mae

from pmdarima.datasets import load_taylor

# src module available here: https://github.com/vcerqueira/blog
from src.tde import time_delay_embedding
from src.ensembles.windowing import WindowLoss
from src.ensembles.ade import Arbitrating
from src.ensembles.oracle import Oracle
from src.plots.barplots import err_barplot
from src.plots.forecasts import train_test_yhat_plot
from src.plots.ts_lineplot_simple import LinePlot

series = load_taylor(as_series=True)
series.index = pd.date_range(end=pd.Timestamp(day=27, month=8, year=2000), periods=len(series), freq='30min')
series.name = 'Series'
series.index.name = 'Index'

series_plot = LinePlot.univariate(series.reset_index(), x_axis_col='Index', y_axis_col='Series')

# train test split
train, test = train_test_split(series, test_size=0.1, shuffle=False)

# ts for supervised learning
train_df = time_delay_embedding(train, n_lags=10, horizon=1).dropna()
test_df = time_delay_embedding(test, n_lags=10, horizon=1).dropna()

# creating the predictors and target variables
X_train, y_train = train_df.drop('Series(t+1)', axis=1), train_df['Series(t+1)']
X_test, y_test = test_df.drop('Series(t+1)', axis=1), test_df['Series(t+1)']

# defining four models composing the ensemble
models = {
    'RF': RandomForestRegressor(),
    'KNN': KNeighborsRegressor(),
    'LASSO': Lasso(),
    'EN': ElasticNetCV(),
    'Ridge': Ridge(),
}

# training and getting predictions
train_forecasts, test_forecasts = {}, {}
for k in models:
    models[k].fit(X_train, y_train)
    train_forecasts[k] = models[k].predict(X_train)
    test_forecasts[k] = models[k].predict(X_test)

# predictions as pandas dataframe
ts_forecasts_df = pd.DataFrame(test_forecasts)
tr_forecasts_df = pd.DataFrame(train_forecasts)

forecasts_df = pd.concat([tr_forecasts_df, ts_forecasts_df], axis=0).reset_index(drop=True)
actual = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

windowing = WindowLoss(lambda_=50)
weights = windowing.get_weights(forecasts_df, actual)
weights = weights.tail(X_test.shape[0]).reset_index(drop=True)

ade = Arbitrating(lambda_=50)
ade.fit(tr_forecasts_df, y_train, X_train)
weights_ade = ade.get_weights(X_test)
weights_ade = weights_ade.tail(X_test.shape[0])

oracle = Oracle()
oracle_weights = oracle.get_weights(forecasts_df, actual)
oracle_weights = pd.DataFrame(oracle_weights, columns=forecasts_df.columns)
oracle_weights = oracle_weights.tail(X_test.shape[0])

# weighting the ensemble dynamically
dynamic = (weights.values * ts_forecasts_df.values).sum(axis=1)
ade_fh = (weights_ade.values * ts_forecasts_df.values).sum(axis=1)
oracle_fh = (oracle_weights.values * ts_forecasts_df.values).sum(axis=1)

# combining the models with static equal weights (average)
static = ts_forecasts_df.mean(axis=1).values

forecasts_df = pd.DataFrame(test_forecasts, index=y_test.index)
forecasts_plot = train_test_yhat_plot(y_train.tail(100), y_test, forecasts_df)

test_forecasts['Windowing'] = dynamic
test_forecasts['Arbitrating'] = ade_fh
test_forecasts['Average'] = static
test_forecasts['Oracle'] = oracle_fh

error = {k: mae(y_test, test_forecasts[k]) for k in test_forecasts}
error.pop('Oracle')
error = pd.Series(error)
error = error.sort_values()

print(error)

plot_err = err_barplot(error)
plot_err.save('error_ensemble_dyn.pdf', width=9, height=4)

weights.index = pd.date_range(end=pd.Timestamp(day=27, month=8, year=2000), periods=weights.shape[0], freq='30min')
weights.index.name = 'Index'
weights_m = weights.reset_index().melt('Index')
weights_m['Type'] = 'Ensemble Weights by Model'

plot = \
    ggplot(weights_m) + \
    aes(x='Index',
        y='value',
        group='variable',
        color='variable') + \
    theme_538(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.125,
          axis_text=element_text(size=10),
          axis_text_x=element_text(angle=20),
          legend_title=element_blank(),
          legend_position='top') + \
    facet_wrap('~Type', scales='free', ncol=1)

plot += geom_line(size=1)
plot = \
    plot + \
    xlab('') + \
    ylab('Weight') + \
    ggtitle('')

plot.save('weights_ts.pdf', width=8.5, height=5)
series_plot.save('plot_taylor.pdf', width=9, height=4)
