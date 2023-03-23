import numpy as np
import pandas as pd
from plotnine import *
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import (Dense,
                          LSTM,
                          TimeDistributed,
                          RepeatVector,
                          Dropout)

from src.tde import (time_delay_embedding,
                     from_3d_to_matrix,
                     from_matrix_to_3d)
from src.utils.log import LogTransformation

N_FEATURES = 1
N_LAGS = 24
HORIZON = 12

######### BASIC CONF #############################################

# Defining the LSTM

# model = Sequential()
# model.add(LSTM(32, activation='relu', input_shape=(N_LAGS, N_FEATURES)))
# model.add(Dropout(.2))
# model.add(RepeatVector(HORIZON))
# model.add(LSTM(16, activation='relu', return_sequences=True))
# model.add(Dropout(.2))
# model.add(TimeDistributed(Dense(N_FEATURES)))
# model.compile(optimizer='adam', loss='mse')
#
# model.summary()
#
# # Training and prediction
#
# X_train, X_valid, Y_train, Y_valid = train_test_split(X_3d, Y_3d, test_size=.2, shuffle=False)
#
# model.fit(X_train, Y_train, epochs=100, validation_data=(X_valid, Y_valid))
#
# preds = model.predict_on_batch(X_valid)
# preds_df = from_3d_to_matrix(preds, Y.columns)

######### Preprocessing #############################################

# https://github.com/vcerqueira/blog/tree/main/data
data = pd.read_csv('data/daily_energy_demand.csv',
                   parse_dates=['Datetime'],
                   index_col='Datetime')

train, test = train_test_split(data, test_size=0.2, shuffle=False)

# plot
plot_df = data.reset_index().melt('Datetime')
plot_df['Type'] = 'Daily power consumption in several USA states'
plot = \
    ggplot(plot_df) + \
    aes(x='Datetime',
        y='np.log(value)',
        group='variable',
        color='variable') + \
    theme_classic(base_family='Palatino', base_size=12) + \
    theme(plot_margin=0.2,
          axis_text=element_text(size=12),
          strip_text=element_text(size=14),
          legend_title=element_blank(),
          legend_position='right')

plot += geom_line()
plot += facet_wrap('~ Type')
plot = \
    plot + \
    xlab('') + \
    ylab('Log power consumption') + \
    ggtitle('')

# Preprocessing

mean_by_series = train.mean()

# mean-scaling
train_scaled = train / mean_by_series
test_scaled = test / mean_by_series

# log transformation
train_scaled_log = LogTransformation.transform(train_scaled)
test_scaled_log = LogTransformation.transform(test_scaled)

# transforming time series for supervised learning
train_by_series, test_by_series = {}, {}
for col in data:
    train_series = train_scaled_log[col]
    test_series = test_scaled_log[col]

    train_series.name = 'Series'
    test_series.name = 'Series'

    train_df = time_delay_embedding(train_series, n_lags=N_LAGS, horizon=HORIZON)
    test_df = time_delay_embedding(test_series, n_lags=N_LAGS, horizon=HORIZON)

    train_by_series[col] = train_df
    test_by_series[col] = test_df

# concatenating all series
train_df = pd.concat(train_by_series, axis=0)
test_df = pd.concat(test_by_series, axis=0)

# defining target (Y) and explanatory variables (X)
predictor_variables = train_df.columns.str.contains('\(t\-|\(t\)')
target_variables = train_df.columns.str.contains('\(t\+')
X_tr = train_df.iloc[:, predictor_variables]
Y_tr = train_df.iloc[:, target_variables]

X_tr_3d = from_matrix_to_3d(X_tr)
Y_tr_3d = from_matrix_to_3d(Y_tr)

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(N_LAGS, N_FEATURES)))
model.add(Dropout(.2))
model.add(RepeatVector(HORIZON))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(Dropout(.2))
model.add(TimeDistributed(Dense(N_FEATURES)))
model.compile(optimizer='adam', loss='mse')

X_train, X_valid, Y_train, Y_valid = \
    train_test_split(X_tr_3d, Y_tr_3d, test_size=.2, shuffle=False)

model.fit(X_train, Y_train,
          validation_data=(X_valid,Y_valid),
          epochs=100)


######### CALLBACKS #############################################

from keras.callbacks import ModelCheckpoint

model_checkpoint = ModelCheckpoint(
    filepath='best_model_weights.h5',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(N_LAGS, N_FEATURES)))
model.add(RepeatVector(HORIZON))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(N_FEATURES)))
model.compile(optimizer='adam', loss='mae')

history = model.fit(X_train, Y_train,
                    epochs=300,
                    validation_data=(X_valid,Y_valid),
                    callbacks=[model_checkpoint])

df = {'Train': history.history['loss'], 'Validation': history.history['val_loss']}
df = pd.DataFrame(df)
df.to_csv('history_nn.csv')

df = pd.read_csv('history_nn.csv', index_col='Unnamed: 0')

df['Validation'].argmin()
df.plot()
df[1:].plot()

# The best model weights are loaded into the model.
model.load_weights('best_model_weights.h5')

# Inference on DAYTON region
test_dayton = test_by_series['DAYTON']

X_ts = test_df.iloc[:, predictor_variables]
Y_ts = test_df.iloc[:, target_variables]
X_ts_3d = from_matrix_to_3d(X_ts)

preds = model.predict_on_batch(X_ts_3d)
preds_df = from_3d_to_matrix(preds, Y_ts.columns)

# reverting log transformation
preds_df = LogTransformation.inverse_transform(preds_df)
# reverting mean scaling
preds_df *= mean_by_series['DAYTON']



Y_ts