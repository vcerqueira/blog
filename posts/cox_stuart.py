import numpy as np
import pandas as pd
from plotnine import *

# RANDOM WALKS

rw = pd.Series(np.cumsum(np.random.choice([-1, 1], size=1000)))
rw_df = rw.reset_index()
rw_df.columns = ['index', 'value']

rw_plot = \
    ggplot(rw_df) + \
    aes(x='index', y='value') + \
    theme_538(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.125,
          axis_text=element_text(size=14),
          axis_text_x=element_blank(),
          legend_title=element_blank(),
          legend_position=None) + \
    geom_line(color="#234f1e", size=1) + \
    labs(x='', y='')

print(rw_plot)

rw100_df = pd.DataFrame([np.cumsum(np.random.choice([-1, 1], size=1000)) for i in range(100)]).T

rw_multiple = ggplot(rw100_df.reset_index().melt('index')) + \
              aes(x='index', y='value', group='variable') + \
              theme_classic(base_family='Palatino', base_size=12) + \
              theme(plot_margin=.125,
                    axis_text=element_blank(),
                    legend_title=element_blank(),
                    legend_position=None) + \
              geom_line(color="#05998c") + \
              labs(x='', y='')

print(rw_multiple)

# UNIT ROOT TESTS

from pmdarima.arima import ndiffs

###

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import seasonal_kendall

# Load data
df = pd.read_csv('my_time_series_data.csv', parse_dates=['date'], index_col='date')

# Decompose data into seasonal, trend, and residual components
decomposition = seasonal_decompose(df, model='additive', period=52)  # 52 for weekly data
seasonal = decomposition.seasonal.dropna()  # Remove missing values
seasonal = seasonal - seasonal.mean()  # Center the data to zero mean

# Calculate the SEAK
seak, pvalue = seasonal_kendall(seasonal)

# Print the results
print('SEAK:', seak)
print('p-value:', pvalue)
