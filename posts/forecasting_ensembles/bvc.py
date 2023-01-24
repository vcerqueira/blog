from plotnine import *
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from pmdarima.datasets import load_sunspots

from src.tde import time_delay_embedding
from src.ensembles.bvc import BiasVarianceCovariance

series = load_sunspots(as_series=True)  # GPL-3
series.index = pd.to_datetime(series.index)

train, test = train_test_split(series, test_size=0.3, shuffle=False, random_state=1)

train_df = time_delay_embedding(train, n_lags=12, horizon=1)
test_df = time_delay_embedding(test, n_lags=12, horizon=1)

# creating the predictors and target variables
target_var = 'Series(t+1)'
X_train, y_train = train_df.drop(target_var, axis=1), train_df[target_var]
X_test, y_test = test_df.drop(target_var, axis=1), test_df[target_var]

# training a random forest and a bagging ensemble with 500 trees each
rf = RandomForestRegressor(n_estimators=100, random_state=1)

rf.fit(X_train, y_train)

# getting predictions from each tree in RF
rf_pred = [tree.predict(X_test) for tree in rf.estimators_]
rf_pred = pd.DataFrame(rf_pred).T

rf_a_bias, rf_a_var, rf_a_cov = BiasVarianceCovariance.get_bvc(rf_pred, y_test.values)

rf_rmse = mean_squared_error(y_test.values, rf_pred.mean(axis=1), squared=False)

df = pd.DataFrame([[rf_a_bias, rf_a_var, rf_a_cov]]).T
df.columns = ['Random Forest']
df['Term'] = ['(Avg Bias)^2', 'Variance', 'Covariance']
dfm = df.melt('Term')

dfm['value'] = np.log(dfm['value'] + 1)
dfm['Term'] = pd.Categorical(dfm['Term'], categories=dfm['Term'])
dfm['Type'] = 'Bias, Variance, and Covariance results for a Random Forest'

plot = ggplot(dfm,
              aes(x='Term', y='value')) + \
       theme_538(base_family='Palatino', base_size=12) + \
       theme(plot_margin=.15,
             axis_text=element_text(size=12),
             legend_title=element_blank(),
             strip_text=element_text(size=14),
             legend_position='top') + \
       geom_bar(position='dodge',
                stat='identity',
                width=0.6,
                fill='#58a63e') + \
       facet_grid('~ Type') + \
       xlab('') + \
       ylab('Log value')

print(plot)

### Sample of the forecasts

yhat = rf_pred.iloc[50:90, :]
train = y_test.iloc[:50]
test = y_test.iloc[50:90]

yhat.index = test.index
yhat.index.name = 'Date'

yhat = yhat.reset_index().melt('Date')

train_df_p = train.reset_index()
train_df_p.columns = ['Date', 'value']
train_df_p['variable'] = 'Train'

test_df_p = test.reset_index()
test_df_p.columns = ['Date', 'value']
test_df_p['variable'] = 'Test'

df = pd.concat([train_df_p, test_df_p, yhat], axis=0)
df['variable'] = pd.Categorical(df['variable'], categories=df['variable'].unique())

avg_yhat = yhat.groupby('Date').mean().reset_index()

plot = \
    ggplot(df) + \
    aes(x='Date',
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
                  mapping=aes(x='Date',
                              y='value',
                              group=1),
                  size=3, color='#0d3aa9')
plot += geom_line(data=train_df_p,
                  mapping=aes(x='Date',
                              y='value',
                              group=1),
                  size=1.3, color='#cf8806')

plot = \
    plot + \
    xlab('') + \
    ylab('') + \
    ggtitle('')
