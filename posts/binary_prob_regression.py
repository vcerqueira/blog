import numpy as np
import pandas as pd
from plotnine import *
from scipy.stats import norm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from src.tde import time_delay_embedding
from src.plots.ts_lineplot_simple import LinePlot

START_DATE = '2022-01-01'
URL = f'https://erddap.marine.ie/erddap/tabledap/IWaveBNetwork.csv?time%2CSignificantWaveHeight&time%3E={START_DATE}T00%3A00%3A00Z&station_id=%22AMETS%20Berth%20B%20Wave%20Buoy%22'

# reading data directly from erdap
data = pd.read_csv(URL, skiprows=[1], parse_dates=['time'])

# setting time to index and getting the target series
series = data.set_index('time')['SignificantWaveHeight']

# transforming data to hourly and from centimeters to meters
series_hourly = series.resample('H').mean() / 100

plot = LinePlot.univariate(series_hourly[500:].reset_index().head(5000),
                           x_axis_col='time',
                           y_axis_col='SignificantWaveHeight',
                           line_color='#0058ab', y_lab='Wave Height (m)')
plot += geom_hline(yintercept=6,
                   linetype='dashed',
                   color='#ba110c',
                   size=2)
print(plot)
plot.save('exceedance_plotts_thr.pdf', width=12, height=6)

train, test = train_test_split(series_hourly, test_size=0.2, shuffle=False)

N_LAGS, HORIZON = 24, 1
THRESHOLD = 6

X_train, Y_train = time_delay_embedding(train, n_lags=N_LAGS, horizon=HORIZON, return_Xy=True)
X_test, Y_test = time_delay_embedding(test, n_lags=N_LAGS, horizon=HORIZON, return_Xy=True)

regression = RandomForestRegressor()
regression.fit(X_train, Y_train)

point_forecasts = regression.predict(X_test)

std = Y_train.std()

exceedance_probability = np.asarray([1 - norm.cdf(THRESHOLD, loc=x_, scale=std)
                                     for x_ in point_forecasts])

# evaluation
y_test_event = (Y_test > THRESHOLD).astype(int)

roc_auc_score(y_test_event, exceedance_probability)

df_num = pd.DataFrame({'Actual': Y_test, 'Point Forecasts': point_forecasts})
plot_forecast = LinePlot.multivariate(data=df_num.reset_index().melt('time'),
                                      x='time', y='value', group='variable',
                                      x_lab='', y_lab='') + \
                scale_color_manual(values=['red', 'blue'])
print(plot_forecast)

df_event = pd.DataFrame({'Event': y_test_event, 'Probability': exceedance_probability})
plot_probs = LinePlot.multivariate(data=df_event.reset_index().melt('time'),
                                   x='time', y='value', group='variable',
                                   x_lab='', y_lab='') + \
             scale_color_manual(values=['red', 'blue'])
print(plot_probs)
plot_probs.save('plot_probs.pdf', width=12, height=6)

#### extra - exceedance via cdf

import numpy as np
import pandas as pd
from scipy.stats import norm

# a random series from the uniform dist.
z = np.random.standard_normal(1000)
# estimating the standard dev.
s = z.std()

# fixing the exceedance threshold
# this is a domain dependent parameter
threshold = 1
# prediction for a given instant
yhat = 0.8

# probability that the actual value exceeds threshold
exceedance_prob = 1 - norm.cdf(threshold, loc=yhat, scale=s)
print(exceedance_prob)

#### extra - cdf curve

from scipy.stats import norm
from plotnine import *

xs = np.linspace(-3, 3, num=100)

df = pd.DataFrame(
    {
        'x': xs,
        'y': [norm.cdf(x) for x in xs]
    }
)

ggplot(data=df,
       mapping=aes(x='x', y='y')) + \
theme_classic(base_family='Palatino', base_size=12) + \
geom_line(size=1.1) + \
labs(x='', y='Probability')
