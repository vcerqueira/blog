import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor
from plotnine import *

from src.tde import time_delay_embedding

wine = pd.read_csv('data/wine_sales.csv', parse_dates=['date'])

# setting date as index
wine.set_index('date', inplace=True)

# you can simulate some data with the following code
# wine = pd.DataFrame(np.random.random((100, 6)),
#            columns=['Fortified','Drywhite','Sweetwhite',
#                      'Red','Rose','Sparkling'])
wine_df = pd.melt(np.log(wine).reset_index(), 'date')

plot = \
    ggplot(wine_df) + \
    aes(x='date',
        y='value',
        group='variable',
        color='variable') + \
    theme_minimal(base_family='Palatino', base_size=12) + \
    theme(plot_margin=0.1,
          axis_text=element_text(size=10),
          axis_text_x=element_text(angle=30),
          legend_title=element_blank(),
          legend_position='top')

plot += geom_line()
plot = \
    plot + \
    xlab('') + \
    ylab('Wine Sales (Log)') + \
    ggtitle('')

plot.save(f'mv_line_plot.pdf', height=5, width=8)

######## Building the correlation matrix

corr_df = pd.melt(wine.corr().reset_index(), 'index')
cats = np.unique(corr_df['index'].unique().tolist() + corr_df['variable'].unique().tolist())

corr_df['index'] = pd.Categorical(corr_df['index'], categories=cats)
corr_df['variable'] = pd.Categorical(corr_df['variable'], categories=cats)
corr_df.columns = ['Wine1', 'Wine2', 'Correlation']

plot = ggplot(corr_df, aes('Wine1', 'Wine2', fill='Correlation')) + \
       geom_tile(aes(width=.95, height=.95)) + \
       theme_minimal(
           base_family='Palatino',
           base_size=12) + \
       theme(
           plot_margin=.2,
           axis_text_x=element_text(angle=45),
           legend_text=element_text(size=8),
           legend_title=element_text(size=8),
           legend_position='top') + \
       xlab('') + \
       ylab('') + \
       scale_fill_gradient2(low='darkorange', mid='white', high='steelblue')

plot.save(f'correlation_matrix_mv.pdf', height=7, width=7)

######## Building the data set

# create data set with lagged features using time delay embedding
wine_ds = []
for col in wine:
    col_df = time_delay_embedding(wine[col], n_lags=12, horizon=6)
    # col_df = col_df.rename(columns=lambda x: re.sub('t', col, x))
    wine_ds.append(col_df)

# concatenating all variables
wine_df = pd.concat(wine_ds, axis=1).dropna()

# defining target (y) and explanatory variables (X)
predictor_variables = wine_df.columns.str.contains('\(t\-')
target_variables = wine_df.columns.str.contains('Sparkling\(t\+')
X = wine_df.iloc[:, predictor_variables]
Y = wine_df.iloc[:, target_variables]

######## Building the forecasting model

# train/test split
X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.3, shuffle=False)

# fitting a linear model
model = RandomForestRegressor()
model.fit(X_tr, Y_tr)

# getting forecasts for the test set
preds = model.predict(X_ts)

# computing MAE error
print(mae(Y_ts, preds))
# 288.13

######## Selecting top features

top_n_features = 10

importance_scores = pd.Series(dict(zip(X_tr.columns, model.feature_importances_)))
top_10_features = importance_scores.sort_values(ascending=False)[:top_n_features]
top_10_features_nm = top_10_features.index

X_tr_top = X_tr[top_10_features_nm]
X_ts_top = X_ts[top_10_features_nm]

# re-fitting a model
model_top_features = RandomForestRegressor()
model_top_features.fit(X_tr_top, Y_tr)

# getting forecasts for the test set
preds_topf = model_top_features.predict(X_ts_top)

# computing MAE error
print(mae(Y_ts, preds_topf))
# 274.36

######### Plotting feature importance

imp_df = importance_scores.sort_values(ascending=True)[-top_n_features:].reset_index()
imp_df.columns = ['Feature', 'Importance']
imp_df['Feature'] = pd.Categorical(imp_df['Feature'], categories=imp_df['Feature'])

plot = ggplot(imp_df, aes(x='Feature', y='Importance')) + \
       geom_bar(fill='steelblue', stat='identity', position='dodge') + \
       theme_classic(
           base_family='Palatino',
           base_size=12) + \
       theme(
           plot_margin=.25,
           axis_text=element_text(size=7),
           axis_title=element_text(size=6),
           legend_text=element_text(size=8),
           legend_title=element_text(size=8),
           legend_position='top') + \
       xlab('') + \
       ylab('Importance') + coord_flip()

plot.save('feature_importance.pdf', height=7, width=5)
