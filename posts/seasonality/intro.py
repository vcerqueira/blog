import numpy as np
import pandas as pd
from plotnine import *

from pmdarima.arima import nsdiffs
from statsmodels.tsa.api import STL

# Deterministic seasonality plot

period = 12
size = 120
beta1 = 0.3
beta2 = 0.6
sin1 = np.asarray([np.sin(2 * np.pi * i / 12) for i in np.arange(1, size + 1)])
cos1 = np.asarray([np.cos(2 * np.pi * i / 12) for i in np.arange(1, size + 1)])

xt = np.cumsum(np.random.normal(scale=0.1, size=size))

yt = xt + beta1 * sin1 + beta2 * cos1 + np.random.normal(scale=0.1, size=size)

series_det = pd.Series(yt)
series_decomp = STL(series_det, period=period).fit()

df = pd.DataFrame(
    {
        'Original series': series_det,
        'Seasonal component': series_decomp.seasonal,
    }
)

df = df.reset_index()
dfm = df.melt('index')

plot = \
    ggplot(dfm) + \
    aes(x='index', y='value') + \
    facet_grid('variable ~.', scales='free') + \
    theme_light(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.125,
          axis_text=element_text(size=11),
          legend_title=element_blank(),
          strip_text=element_text(size=12, color='#281e5d'),
          legend_position='none') + \
    labs(x='Time step',
         y='value') + \
    geom_line(color='#281e5d', size=1)

print(plot)

# Stochastic stationary seasonality

period = 12
size = 120
beta1 = np.linspace(-.6, .3, num=size)
beta2 = np.linspace(.6, -.3, num=size)
sin1 = np.asarray([np.sin(2 * np.pi * i / 12) for i in np.arange(1, size + 1)])
cos1 = np.asarray([np.cos(2 * np.pi * i / 12) for i in np.arange(1, size + 1)])

xt = np.cumsum(np.random.normal(scale=0.1, size=size))

yt = xt + beta1 * sin1 + beta2 * cos1 + np.random.normal(scale=0.1, size=size)

series_stoc = pd.Series(yt)

series_decomp = STL(series_stoc, period=period).fit()

df = pd.DataFrame(
    {
        'Original series': series_stoc,
        'Seasonal component': series_decomp.seasonal,
    }
)

df = df.reset_index()
dfm = df.melt('index')

plot = \
    ggplot(dfm) + \
    aes(x='index', y='value') + \
    facet_grid('variable ~.', scales='free') + \
    theme_light(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.125,
          axis_text=element_text(size=11),
          legend_title=element_blank(),
          strip_text=element_text(size=12, color='#2832c2'),
          legend_position='none') + \
    labs(x='Time step',
         y='value') + \
    geom_line(color='#2832c2', size=1)

print(plot)


# Seasonal strength

def seasonal_strength(series: pd.Series) -> float:
    series_decomp = STL(series, period=period).fit()

    resid_seas_var = (series_decomp.resid + series_decomp.seasonal).var()
    resid_var = series_decomp.resid.var()

    result = 1 - (resid_var / resid_seas_var)

    return result


seasonal_strength(series_det)
seasonal_strength(series_stoc)

# nsdiffs
nsdiffs(x=series_det, m=period, test='ch')
nsdiffs(x=series_det, m=period, test='ocsb')

nsdiffs(x=series_stoc, m=period, test='ch')
nsdiffs(x=series_stoc, m=period, test='ocsb')
