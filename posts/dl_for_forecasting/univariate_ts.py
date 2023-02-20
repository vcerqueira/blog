import pandas as pd

from src.tde import (time_delay_embedding,
                     from_3d_to_matrix,
                     from_matrix_to_3d)

# https://github.com/vcerqueira/blog/tree/main/data
data = pd.read_csv('data/wine_sales.csv', parse_dates=['date'])
data.set_index('date', inplace=True)

N_FEATURES = 1
N_LAGS = 3
HORIZON = 2

series = data['Sparkling']

"""
from src.plots.ts_lineplot_simple import LinePlot
from plotnine import *

series_df = series.reset_index()
series_df['Type'] = 'Number of sales of sparkling wine'
plot = LinePlot.univariate(series_df,
                           x_axis_col='date',
                           y_axis_col='Sparkling',
                           line_color='#0058ab', y_lab='No. of Sales')
plot += facet_wrap('~ Type',nrow=1)
print(plot)
plot.save('univ.pdf', height=4, width=8)
"""

"""
series_mat = time_delay_embedding(series, n_lags=N_LAGS, horizon=HORIZON)
"""

X, Y = time_delay_embedding(series, n_lags=N_LAGS, horizon=HORIZON, return_Xy=True)

"""
from sklearn.linear_model import RidgeCV
model = RidgeCV()
model.fit(X, Y)
"""



X_3d = from_matrix_to_3d(X)
Y_3d = from_matrix_to_3d(Y)

# Defining the LSTM ##################################################

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import (Dense,
                          LSTM,
                          TimeDistributed,
                          RepeatVector,
                          Dropout)

model = Sequential()
model.add(LSTM(8, activation='relu', input_shape=(N_LAGS, N_FEATURES)))
model.add(RepeatVector(HORIZON))
model.add(LSTM(4, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(N_FEATURES)))
model.compile(optimizer='adam', loss='mse')

model.summary()


######################################################################

"""
model = Sequential()
model.add(LSTM(units=16, activation='relu', input_shape=(N_LAGS, N_FEATURES)))
model.add(Dropout(rate=.25))
model.add(Dense(units=HORIZON))

model.compile(optimizer='adam', loss='mse')
"""

X_train, X_valid, Y_train, Y_valid = train_test_split(X_3d, Y_3d, test_size=.2, shuffle=False)

model.fit(X_train, Y_train, epochs=100, validation_data=(X_valid, Y_valid))

preds = model.predict_on_batch(X_valid)

preds_df = from_3d_to_matrix(preds, Y.columns)
# preds_df = pd.DataFrame(preds, columns=Y.columns)
y_valid_df = from_3d_to_matrix(Y_valid, Y.columns)
