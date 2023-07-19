import numpy as np
import pandas as pd
from plotnine import *

# time series w stochastic stationary seasonality

period1, period2 = 24, 24 * 7
size = 500
beta1 = np.linspace(-.6, .3, num=size)
beta2 = np.linspace(.6, -.3, num=size)
sin1 = np.asarray([np.sin(2 * np.pi * i / period1) for i in np.arange(1, size + 1)])
sin2 = np.asarray([np.sin(2 * np.pi * i / period2) for i in np.arange(1, size + 1)])
cos1 = np.asarray([np.cos(2 * np.pi * i / period1) for i in np.arange(1, size + 1)])
cos2 = np.asarray([np.cos(2 * np.pi * i / period2) for i in np.arange(1, size + 1)])

xt = np.cumsum(np.random.normal(scale=0.1, size=size))
noise = np.random.normal(scale=0.1, size=size)

yt = xt + beta1 * sin1 + beta2 * cos1 + sin2 + cos2 + noise

ind = pd.date_range(end=pd.Timestamp('2023-07-10'), periods=size, freq='H')
yt = pd.Series(yt, index=ind)
yt.name = 'Series'
yt.index.name = 'Date'

# yt.plot()

####################################################
######## MSTL

dfr = yt.reset_index()

plot = \
    ggplot(dfr) + \
    aes(x='Date', y='Series') + \
    theme_538(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.125,
          axis_text=element_text(size=11),
          legend_title=element_blank(),
          legend_text=element_text(size=15),
          panel_background=element_rect(fill='white'),
          plot_background=element_rect(fill='white'),
          strip_background=element_rect(fill='white'),
          legend_background=element_rect(fill='white'),
          legend_position='top') + \
    labs(x='', y='') + \
    geom_line(size=1)

from statsmodels.tsa.seasonal import MSTL

decomp = MSTL(endog=yt, periods=(period1, period2)).fit()

df = {
    'Original': yt,
    'Trend': decomp.trend,
    'Daily Seas': decomp.seasonal['seasonal_24'],
    'Weekly Seas': decomp.seasonal['seasonal_168'],
    'Residuals': decomp.resid,
}

df = pd.concat(df, axis=1)

dfr = df.reset_index()
dfm = dfr.melt('Date')
dfm['variable'] = pd.Categorical(dfm['variable'],
                                 categories=['Original',
                                             'Trend',
                                             'Daily Seas',
                                             'Weekly Seas',
                                             'Residuals'])

plot = \
    ggplot(dfm) + \
    aes(x='Date', y='value') + \
    facet_grid('variable ~.', scales='free') + \
    theme_538(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.125,
          axis_text_x=element_text(size=8),
          legend_title=element_blank(),
          panel_background=element_rect(fill='white'),
          plot_background=element_rect(fill='white'),
          strip_background=element_rect(fill='white'),
          legend_background=element_rect(fill='white'),
          strip_text=element_text(size=12, color='black'),
          legend_position='none') + \
    labs(x='', y='value') + \
    geom_line(color='black', size=1.2)

# print(plot)

################################################################
# Fourier series with multiple periods
########################################################

from sktime.transformations.series.fourier import FourierFeatures

fourier = FourierFeatures(sp_list=[period1, period2],
                          fourier_terms_list=[4, 2],
                          keep_original_columns=False)

fourier_feats = fourier.fit_transform(yt)

# KTR model ################################################################
############################################################################

from orbit.models import KTR
from orbit.diagnostics.plot import plot_predicted_data, plot_predicted_components
from sklearn.model_selection import train_test_split

df = yt.reset_index()

train, test = train_test_split(df, shuffle=False, test_size=100)

ktr_with_seas = KTR(
    response_col='Series',
    date_col='Date',
    seed=1,
    seasonality=[24, 24 * 7],
    estimator='pyro-svi',
    n_bootstrap_draws=1e4,
    # pyro training config
    num_steps=301,
    message=100,
)

ktr_with_seas.fit(train)
predicted_df = ktr_with_seas.predict(df=df, decompose=True)

_ = plot_predicted_data(training_actual_df=train,
                        predicted_df=predicted_df,
                        date_col='Date',
                        actual_col='Series',
                        test_actual_df=test,
                        markersize=10, lw=.5)

_ = plot_predicted_components(predicted_df=predicted_df,
                              date_col='Date',
                              plot_components=['trend',
                                               'seasonality_24',
                                               'seasonality_168'])
