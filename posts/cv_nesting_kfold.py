import numpy as np
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import (GridSearchCV,
                                     KFold,
                                     train_test_split)

# creating a dummy data set
X, y = make_regression(n_samples=100)

# outer split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, shuffle=False, test_size=0.2)

# inner cv procedure
cv = KFold(n_splits=5)

# defining the search space
# a simple optimization of the number of trees of a RF
model = RandomForestRegressor()
param_search = {'n_estimators': [10, 50, 100]}

# applying CV with a gridsearch on the training data
gs = GridSearchCV(estimator=model,
                  cv=cv,
                  param_grid=param_search)

gs.fit(X_train, y_train)

# re-training the best approach for testing
chosen_model = RandomForestRegressor(**gs.best_params_)
chosen_model.fit(X_train, y_train)

# inference on test set and evaluation
preds = chosen_model.predict(X_test)
estimated_performance = r2_score(y_test, preds)

# final model for deployment
final_model = RandomForestRegressor(**gs.best_params_)
final_model.fit(X, y)


## plots


from plotnine import *

N_SPLITS = 5

X, y = make_regression(n_samples=30)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, shuffle=False, test_size=0.2)

cv = KFold(n_splits=5, shuffle=False)

development = np.arange(0, X.shape[0])
dev_df = pd.DataFrame(development)
dev_df.columns = ['x']
dev_df['part'] = 'All Available Data'
dev_df['size'] = 1.75
dev_df['y'] = N_SPLITS + 2

training_data = np.arange(0, X_train.shape[0])
train_df = pd.DataFrame(training_data)
train_df.columns = ['x']
train_df['part'] = 'Train'
train_df['size'] = 1.75
train_df['y'] = N_SPLITS + 1

test_data = np.arange(X_train.shape[0]+1, X_train.shape[0]+X_test.shape[0])
test_df = pd.DataFrame(test_data)
test_df.columns = ['x']
test_df['part'] = 'Test'
test_df['size'] = 1.75
test_df['y'] = N_SPLITS + 1

future_data = np.arange(X.shape[0]+1, X.shape[0] + 5)
fut_df = pd.DataFrame(future_data)
fut_df.columns = ['x']
fut_df['part'] = 'Future Data'
fut_df['size'] = 1.75
fut_df['y'] = N_SPLITS + 2

segments_data = []
for i, (tr, ts) in enumerate(cv.split(X_train, y_train)):
    tr_df = pd.DataFrame(tr)
    tr_df.columns = ['x']
    tr_df['part'] = 'CV - Train'
    ts_df = pd.DataFrame(ts)
    ts_df.columns = ['x']
    ts_df['part'] = 'CV - Validation'

    df = pd.concat([tr_df, ts_df], axis=0)
    df['size'] = 1.75
    unused = np.setdiff1d(np.arange(X_train.shape[0]), df['x'].values)

    df['y'] = i

    segments_data.append(df)

segments_data.append(dev_df)
segments_data.append(train_df)
segments_data.append(test_df)
segments_data.append(fut_df)

segments_df = pd.concat(segments_data, axis=0)
segments_df['part'] = pd.Categorical(segments_df['part'], ['CV - Validation',
                                                           'CV - Train',

                                                           'Train',
                                                           'Test',
                                                           'All Available Data',
                                                           'Future Data'])
segments_df = segments_df.sort_values('part')

segments_df['y'] += 2

plot = \
    ggplot(segments_df) + \
    aes(x='x', y='y', color='part', size='size') + \
    theme_538(base_family='Palatino', base_size=10) + \
    theme(plot_margin=.2,
          axis_text=element_blank(),
          legend_title=element_blank(),
          legend_position='right') + \
    geom_point(shape="s") + \
    xlab('') + \
    ylab('') + \
    ggtitle('') + \
    scale_y_reverse() + \
    scale_color_manual(values=['#f0a800', '#a9d39e', '#58a63e',
                               '#cf8806', '#2b5c0e', '#bc544b']) + \
    guides(size=None)  # + \
# scale_y_continuous(labels=lambda lst: list(reversed([str(int(x)) if x != 6 else 'All' for x in lst])))
# scale_y_continuous(labels=lambda lst: [str(int(x)) if x != 6 else 'All' for x in lst])

plot += geom_vline(xintercept=24,
                   linetype='dashed',
                   color='#b6dea8',
                   size=1.1)

plot += geom_vline(xintercept=30,
                   linetype='dashed',
                   color='#b6dea8',
                   size=1.1)


plot += geom_text(mapping=aes(x=30, y=7),
                  label='Re-train',
                  color='black',
                  size=11, angle=90)

plot += geom_text(mapping=aes(x=24, y=7),
                  label='Re-train',
                  color='black',
                  size=11, angle=90)

plot += geom_text(mapping=aes(x=15, y=7),
                  label='After Cross-validation',
                  color='black',
                  size=11, angle=0)

plot += geom_text(mapping=aes(x=15, y=1),
                  label='Cross-validation',
                  color='black',
                  size=11, angle=0)


print(plot)


plot.save('cv_nested.pdf', width=9, height=7)
