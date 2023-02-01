import numpy as np
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold

X, y = make_regression(n_samples=100)

cv = KFold(n_splits=5)

model = RandomForestRegressor()
param_search = {'n_estimators': [10, 50, 100]}

gs = GridSearchCV(estimator=model, cv=cv, refit=True, param_grid=param_search)
gs.fit(X, y)

# best_model = RandomForestRegressor(**gs.best_params_)

## plots


from plotnine import *

N_SPLITS = 5

X, y = make_regression(n_samples=30)

cv = KFold(n_splits=5, shuffle=False)

development = np.arange(0, X.shape[0])
dev_df = pd.DataFrame(development)
dev_df.columns = ['x']
dev_df['part'] = 'Available Data'
dev_df['size'] = 1.75
dev_df['y'] = N_SPLITS + 1

future_data = np.arange(X.shape[0] + 1, X.shape[0] + 4)
fut_df = pd.DataFrame(future_data)
fut_df.columns = ['x']
fut_df['part'] = 'Future Data'
fut_df['size'] = 1.75
fut_df['y'] = N_SPLITS + 1

segments_data = []
for i, (tr, ts) in enumerate(cv.split(X, y)):
    tr_df = pd.DataFrame(tr)
    tr_df.columns = ['x']
    tr_df['part'] = 'CV - Training'
    ts_df = pd.DataFrame(ts)
    ts_df.columns = ['x']
    ts_df['part'] = 'CV - Validation'

    df = pd.concat([tr_df, ts_df], axis=0)
    df['size'] = 1.75
    unused = np.setdiff1d(np.arange(X.shape[0]), df['x'].values)

    unused_df = pd.DataFrame(unused)
    unused_df.columns = ['x']
    unused_df['part'] = 'Unused'
    unused_df['size'] = 1
    df = pd.concat([df, unused_df], axis=0)

    df['y'] = i

    segments_data.append(df)

segments_data.append(dev_df)
segments_data.append(fut_df)

segments_df = pd.concat(segments_data, axis=0)
segments_df['part'] = pd.Categorical(segments_df['part'], ['CV - Training',
                                                           'CV - Validation',
                                                           'Available Data',
                                                           'Future Data'])
segments_df = segments_df.sort_values('part')

segments_df['y'] += 1

plot = \
    ggplot(segments_df) + \
    aes(x='x', y='y', color='part', size='size') + \
    theme_538(base_family='Palatino', base_size=10) + \
    theme(plot_margin=.25,
          axis_text=element_blank(),
          legend_title=element_blank(),
          legend_position='right') + \
    geom_point(shape="s") + \
    xlab('') + \
    ylab('') + \
    ggtitle('') + \
    scale_y_reverse() + \
    scale_color_manual(values=['#58a63e',
                               '#cf8806',
                               '#2b5c0e',
                               '#bc544b']) + \
    guides(size=None)

plot += geom_vline(xintercept=30,
                   linetype='dashed',
                   color='#b6dea8',
                   size=1.1)

plot += geom_text(mapping=aes(x=30, y=8),
                  label='Re-train best model \n with Available Data',
                  color='black',
                  size=11)

plot += geom_text(mapping=aes(x=15, y=6),
                  label='After Cross-validation',
                  color='black',
                  size=11, angle=0)

plot += geom_text(mapping=aes(x=15, y=0),
                  label='Cross-validation',
                  color='black',
                  size=11, angle=0)

print(plot)

plot.save('cv_retrain.pdf', width=11, height=7.5)

## plot simple

segments_data = []
for i, (tr, ts) in enumerate(cv.split(X, y)):
    tr_df = pd.DataFrame(tr)
    tr_df.columns = ['x']
    tr_df['part'] = 'Training'
    ts_df = pd.DataFrame(ts)
    ts_df.columns = ['x']
    ts_df['part'] = 'Validation'

    df = pd.concat([tr_df, ts_df], axis=0)
    df['size'] = 1.5
    unused = np.setdiff1d(np.arange(X.shape[0]), df['x'].values)

    df['y'] = i

    segments_data.append(df)

segments_df = pd.concat(segments_data, axis=0)
segments_df['part'] = pd.Categorical(segments_df['part'], ['Training', 'Validation'])
segments_df = segments_df.sort_values('part')

segments_df['y'] += 1

plot = \
    ggplot(segments_df) + \
    aes(x='x', y='y', color='part', size='size') + \
    theme_538(base_family='Palatino', base_size=10) + \
    theme(plot_margin=.25,
          axis_text=element_text(size=12),
          axis_title=element_text(size=12),
          axis_text_x=element_blank(),
          legend_title=element_blank(),
          legend_position='right') + \
    geom_point(shape="s") + \
    xlab('') + \
    ylab('CV Iteration') + \
    ggtitle(f'Data Partitions with 5-fold CV') + \
    scale_y_reverse() + \
    scale_color_manual(values=['#58a63e', '#cf8806']) + \
    guides(size=None)

print(plot)

plot.save('5fcv.pdf', width=10, height=6)