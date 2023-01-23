from plotnine import *
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNetCV

from pmdarima.datasets import load_taylor

# src module available here: https://github.com/vcerqueira/blog
from src.tde import time_delay_embedding
from src.plots.forecasts import train_test_yhat_plot
from src.ensembles.opera_r import Opera

series = load_taylor(as_series=True)
series.index = pd.date_range(end=pd.Timestamp(day=27, month=8, year=2000), periods=len(series), freq='30min')
series.name = 'Series'
series.index.name = 'Index'

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
test_forecasts = {}
for k in models:
    models[k].fit(X_train, y_train)
    test_forecasts[k] = models[k].predict(X_test)

# predictions as pandas dataframe
forecasts_df = pd.DataFrame(test_forecasts, index=y_test.index)

opera = Opera('MLpol')
opera.compute_weights(forecasts_df, y_test)

ensemble = (opera.weights.values * forecasts_df).sum(axis=1)

forecasts_plot = train_test_yhat_plot(y_test.head(50),
                                      y_test.iloc[50:100],
                                      forecasts_df.iloc[50:100, :])

opera.weights.index = y_test.index
opera.weights.reset_index(inplace=True)

weights_df = opera.weights.melt('Index')

plot1 = ggplot(weights_df, aes(x=1, y='value', fill='variable')) + \
        theme_538(base_family='Palatino', base_size=12) + \
        theme(plot_margin=.15,
              axis_text=element_text(size=12),
              axis_text_x=element_blank(),
              legend_title=element_blank(),
              legend_position=None) + \
        geom_boxplot(width=0.5,
                     show_legend=False) + \
        xlab('') + \
        ylab('Weight Distribution') + \
        facet_wrap('~ variable', nrow=1, scales='free_x')

plot2 = ggplot(weights_df,
               aes(x='Index', y='value', fill='variable')) + \
        theme_538(base_family='Palatino', base_size=12) + \
        theme(plot_margin=.15,
              axis_text=element_text(size=12),
              axis_text_x=element_text(size=10),
              legend_title=element_blank(),
              legend_position='top') + \
        geom_area() + \
        xlab('') + \
        ylab('Weight')
