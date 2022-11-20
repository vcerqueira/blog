import pandas as pd

from plotnine import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from pmdarima.datasets import load_sunspots

from src.tde import time_delay_embedding

# loading the sunspots time series
series = load_sunspots(as_series=True).diff()

# applying time delay embedding
ts = time_delay_embedding(series=series, n_lags=3, horizon=1)

# splitting target variable from explanatory variables
target_columns = ts.columns.str.contains('\+')
X = ts.iloc[:, ~target_columns]
y = ts['Series(t+1)']

# train/test split
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.33, shuffle=False)

# adding the mean as features
X_tr['mean'] = X_tr.mean(axis=1)
X_ts['mean'] = X_ts.mean(axis=1)

# fitting a random forest
model = RandomForestRegressor()
model.fit(X_tr, y_tr)

# making predictions
predictions = model.predict(X_ts)

# computing error
mae(y_ts, predictions)
# 13.23


############################## Time series plot

df = load_sunspots(as_series=True).reset_index()
df.columns = ['Date', 'Sunspots']
df['Date'] = pd.to_datetime(df['Date'])

plot = \
    ggplot(df) + \
    aes(x='Date', y='Sunspots', group=1) + \
    theme_minimal(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.15,
          axis_text=element_text(size=10),
          legend_title=element_blank(),
          legend_position=None)

plot += geom_line(color='#cf8806')

plot = \
    plot + \
    xlab('') + \
    ylab('Sunspots') + \
    ggtitle('')

plot.save('sunspots_series.pdf', height=3, width=6)

############################## Importance plot

importance_scores = pd.Series(dict(zip(X_tr.columns, model.feature_importances_)))

imp_df = importance_scores.sort_values(ascending=False).reset_index()
imp_df.columns = ['Feature', 'Importance']
imp_df['Feature'] = pd.Categorical(imp_df['Feature'], categories=imp_df['Feature'])

plot = ggplot(imp_df, aes(x='Feature', y='Importance')) + \
       geom_bar(fill='#cf8806', stat='identity', position='dodge') + \
       theme_classic(
           base_family='Palatino',
           base_size=12) + \
       theme(
           plot_margin=.2,
           axis_text=element_text(size=10),
           axis_title=element_text(size=8),
           legend_text=element_text(size=8),
           legend_title=element_text(size=8),
           legend_position='top') + \
       xlab('') + \
       ylab('Importance') + coord_flip()

plot.save('feature_importance_tde_intro.pdf', height=3, width=3.5)
