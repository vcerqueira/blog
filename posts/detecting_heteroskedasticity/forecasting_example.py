from plotnine import *

import pandas as pd
import numpy as np
from pmdarima.datasets import load_airpassengers
from pmdarima.arima import auto_arima
from sklearn.model_selection import train_test_split

series = load_airpassengers(True)

# leaving the last 12 points for testing
train, test = train_test_split(series, test_size=12, shuffle=False)
# stabilizing the variance in the train
log_train = np.log(train)

# building an arima model, m is the seasonal period (monthly)
mod = auto_arima(log_train, seasonal=True, m=12)

# getting the log forecasts
log_forecasts = mod.predict(12)

# reverting the forecasts
forecasts = np.exp(log_forecasts)

### plotting

train_df = train.reset_index()
train_df['part'] = 'Train'
log_forecasts_df = log_forecasts.reset_index()
log_forecasts_df['part'] = 'Log Forecasts'
forecasts_df = forecasts.reset_index()
forecasts_df['part'] = 'Forecasts'

df = pd.concat([train_df, log_forecasts_df, forecasts_df], axis=0)
df = df.reset_index(drop=True)
df.columns = ['Index', 'Value', 'Part']

PISTACHIO_HARD = '#58a63e'
PISTACHIO_BLACK = '#2b5c0e'
PISTACHIO_MID = '#a9d39e'
PISTACHIO_SOFT = '#b6dea8'
PISTACHIO_FILL = '#edf7ea'

plot = \
    ggplot(df) + \
    aes(x='Index',
        y='Value',
        group='Part',
        color='Part') + \
    theme_minimal(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.15,
          legend_title=element_blank(),
          legend_position='top') + \
    scale_color_manual(values=[PISTACHIO_BLACK,
                               PISTACHIO_MID,
                               PISTACHIO_HARD])

plot += geom_line(size=1)

plot = \
    plot + \
    xlab('') + \
    ylab('') + \
    ggtitle('')


plot.save('forecasts.pdf', width=8, height=3)
