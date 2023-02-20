import pandas as pd
from plotnine import *

from src.tde import (time_delay_embedding,
                     from_3d_to_matrix,
                     from_matrix_to_3d)

# https://github.com/vcerqueira/blog/tree/main/data
data = pd.read_csv('data/wine_sales.csv', parse_dates=['date'])
data.set_index('date', inplace=True)

N_FEATURES = data.shape[1]
N_LAGS = 3
HORIZON = 2

plot_df = data.reset_index().melt('date')
plot_df['Type'] = 'Sales of different types of wine'
plot = \
    ggplot(plot_df) + \
    aes(x='date',
        y='np.log(value)',
        group='variable',
        color='variable') + \
    theme_538(base_family='Palatino', base_size=12) + \
    theme(plot_margin=0.2,
          axis_text=element_text(size=10),
          axis_text_x=element_text(angle=0, size=8),
          legend_title=element_blank(),
          legend_position='right')

plot += geom_line()
plot += facet_wrap('~ Type')
plot = \
    plot + \
    xlab('') + \
    ylab('Wine Sales (Log)') + \
    ggtitle('')

# print(plot)

# plot.save('mv_line_plot.pdf', height=5, width=8)


# transforming each variable into a matrix format
mat_by_variable = []
for col in data:
    col_df = time_delay_embedding(data[col], n_lags=N_LAGS, horizon=HORIZON)
    mat_by_variable.append(col_df)

# concatenating all variables
mat_df = pd.concat(mat_by_variable, axis=1).dropna()

# target_var = 'Sparkling'
# defining target (Y) and explanatory variables (X)
predictor_variables = mat_df.columns.str.contains('\(t\-|\(t\)')
# target_variables = mat_df.columns.str.contains(f'{target_var}\(t\+')
target_variables = mat_df.columns.str.contains('\(t\+')
X = mat_df.iloc[:, predictor_variables]
Y = mat_df.iloc[:, target_variables]

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
model.add(Dropout(.2))
model.add(RepeatVector(HORIZON))
model.add(LSTM(4, activation='relu', return_sequences=True))
model.add(Dropout(.2))
model.add(TimeDistributed(Dense(N_FEATURES)))
model.compile(optimizer='adam', loss='mse')
model.summary()

######################################################################


X_train, X_valid, Y_train, Y_valid = train_test_split(X_3d, Y_3d, test_size=.2, shuffle=False)

model.fit(X_train, Y_train, epochs=500, validation_data=(X_valid, Y_valid))

preds = model.predict_on_batch(X_valid)

preds_df = from_3d_to_matrix(preds, Y.columns)
print(preds_df)
# preds_df = pd.DataFrame(preds, columns=Y.columns)


# plotting forecasts
y_valid_df = data.tail(preds.shape[0])
y_train_df = data.tail(preds.shape[0] + preds.shape[0]).head(preds.shape[0])
yhat = preds_df[[f'{c}(t+1)' for c in data.columns]]
yhat.columns = data.columns
yhat.index = y_valid_df.index
yhat.index.name = 'Date'
yhat['Part'] = 'Forecasts'
y_train_df.index.name='Date'

yhat = yhat.reset_index().melt('Date')
# yhat['size'] = .3

train_df_p = y_train_df.reset_index().melt('Date')
train_df_p['Part'] = 'Train'

test_df_p = y_valid_df.reset_index().melt('Date')
test_df_p['Part'] = 'Test'

df = pd.concat([train_df_p, yhat], axis=0)
#df = pd.concat([train_df_p, test_df_p, yhat], axis=0)
df['Part'] = pd.Categorical(df['Part'], categories=df['Part'].unique())

plot = \
    ggplot(df) + \
    aes(x='Date',
        y='value',
        group='variable',
        color='variable') + \
    theme_538(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.2,
          axis_text=element_text(size=10),
          axis_text_x=element_text(),
          legend_title=element_blank(),
          legend_position='right')  # + guides(size=False)

plot += geom_line(size=1)
plot += geom_vline(xintercept=pd.Timestamp('1992-07-15'),
                   linetype='dashed',
                   color='#cf8806',
                   size=1.1)

plot += geom_text(mapping=aes(x=pd.Timestamp('1993-01-01'), y=6475),
                  label='Forecasts',
                  color='black',
                  size=13, angle=0)

plot += geom_text(mapping=aes(x=pd.Timestamp('1992-02-01'), y=6500),
                  label='Training',
                  color='black',
                  size=13, angle=0)

plot = \
    plot + \
    xlab('') + \
    ylab('No. of sales') + \
    ggtitle('')

plot.save(f'mv_line_forecasts.pdf', height=6, width=11)

