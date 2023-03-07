import numpy as np
import pandas as pd
from plotnine import *
from numerize import numerize
from statsmodels.tsa.arima.model import ARIMA

series = pd.read_csv('data/gdp-countries.csv')['United States'].dropna()
series.index = pd.date_range(start='12/31/1959', periods=len(series), freq='Y')

series_df = series.reset_index()
series_df.columns = ['Date', 'GDP']
series_df['log GDP'] = np.log(series_df['GDP'])

# GDP PLOTS

gdp_uv_plot = \
    ggplot(series_df) + \
    aes(x='Date', y='GDP', group=1) + \
    theme_classic(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.125,
          axis_text=element_text(size=14),
          legend_title=element_blank(),
          legend_position=None) + \
    geom_line(color="#234f1e", size=2) + \
    labs(x='', y='GDP') + \
    scale_y_continuous(labels=lambda lst: [numerize.numerize(x) for x in lst])


gdp_facet_plot = \
    ggplot(series_df.melt('Date')) + \
    aes(x='Date', y='value') + \
    facet_grid('variable ~.', scales='free') + \
    theme_classic(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.125,
          axis_text=element_text(size=14),
          legend_title=element_blank(),
          strip_text=element_text(size=13),
          legend_position='none') + \
    scale_y_continuous(labels=lambda lst: [numerize.numerize(x) for x in lst]) + \
    geom_line(color='#234f1e', size=2) + \
    labs(x='', y='')

gdp_uv_plot.save('plot_gdp_usa.pdf', width=12, height=5)
gdp_facet_plot.save('plot_gdpfacet_usa.pdf', width=12, height=7)

# LINEAR DETERMINISTIC TREND MODEL

log_gdp = np.log(series)

linear_trend = np.arange(1, len(log_gdp) + 1)
model = ARIMA(endog=log_gdp, order=(1, 0, 0), exog=linear_trend)
result = model.fit()

# RANDOM WALKS

rw = pd.Series(np.cumsum(np.random.choice([-1, 1], size=1000)))
rw_df = rw.reset_index()
rw_df.columns = ['index','value']

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
ndiffs(log_gdp, test='adf')

diff_log_gdp = log_gdp.diff().diff()
diff_gdp_df = diff_log_gdp.reset_index()
diff_gdp_df.columns = ['Date','data']

diffgdp_plot = \
    ggplot(diff_gdp_df) + \
    aes(x='Date', y='data') + \
    theme_classic(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.125,
          axis_text=element_text(size=14),
          axis_text_x=element_blank(),
          legend_title=element_blank(),
          legend_position=None) + \
    geom_line(color="#234f1e", size=1) + \
    labs(x='', y='')

print(diffgdp_plot)


# UNIT ROOT TESTING

from statsmodels.tsa.stattools import adfuller, kpss

pvalue_adf = adfuller(x=log_gdp, regression='ct')[1]
pvalue_kpss = kpss(x=log_gdp, regression='ct')[1]

print(pvalue_adf)
print(pvalue_kpss)

pvalue_adf = adfuller(x=rw)[1]
pvalue_kpss = kpss(x=rw)[1]

print(pvalue_adf)
print(pvalue_kpss)

# DIFFERENCING


#

#
# from pmdarima.datasets import load_airpassengers
# from pmdarima.arima import ARIMA
#
# series = load_airpassengers(True)
#
# d_model = ARIMA(order=(1, 0, 1), trend='ct')
# d_model.fit(y=series)
# d_forecasts, d_forecasting_intervals =  d_model.predict(n_periods=12, return_conf_int=True)
#
# s_model = ARIMA(order=(1, 1, 1))
# s_model.fit(y=series)
# s_forecasts, s_forecasting_intervals =  s_model.predict(n_periods=12, return_conf_int=True)
#
#


###
#
# import pandas as pd
# from plotnine import *
#
# from pmdarima.datasets import load_airpassengers
#
# from src.plots.ts_lineplot_simple import LinePlot
#
# series = load_airpassengers(True)
# series.index = pd.date_range(start=pd.Timestamp('1949-01-01'), periods=len(series), freq='MS')
#
# series_df = series.reset_index()
# series_df.columns = ['Date', 'value']
# series_df['Type'] = 'Time series plot with overlayed trend'
#
# plot = LinePlot.univariate(series_df,
#                            x_axis_col='Date',
#                            y_axis_col='value',
#                            line_color='#0058ab',
#                            y_lab='No. of Passengers',
#                            add_smooth=True)
# plot += facet_wrap('~ Type', nrow=1)
# plot += theme(strip_text=element_text(size=14))
# print(plot)
#
# series_df = log_gdp.reset_index()
# series_df.columns = ['Date', 'value']
# plot_trend = LinePlot.univariate(series_df,
#                                  x_axis_col='Date',
#                                  y_axis_col='value',
#                                  line_color='#0058ab',
#                                  y_lab='GDP',
#                                  add_smooth=False)
#
# ##
