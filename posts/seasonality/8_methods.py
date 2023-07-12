import numpy as np
import pandas as pd
from plotnine import *

from pmdarima.arima import nsdiffs
from statsmodels.tsa.api import STL

from src.seasonality import seasonal_strength, AutoCorrelation

# time series w stochastic stationary seasonality

period = 12
size = 120
beta1 = np.linspace(-.6, .3, num=size)
beta2 = np.linspace(.6, -.3, num=size)
sin1 = np.asarray([np.sin(2 * np.pi * i / 12) for i in np.arange(1, size + 1)])
cos1 = np.asarray([np.cos(2 * np.pi * i / 12) for i in np.arange(1, size + 1)])

xt = np.cumsum(np.random.normal(scale=0.1, size=size))

yt = xt + beta1 * sin1 + beta2 * cos1 + np.random.normal(scale=0.1, size=size)

ind = pd.date_range(end=pd.Timestamp('2023-07-10'), periods=120, freq='M')
yt = pd.Series(yt, index=ind)
yt.name = 'Series'
yt.index.name = 'Date'

#
# # Deterministic seasonality plot
#
# period = 12
# size = 120
# beta1 = 0.3
# beta2 = 0.6
# sin1 = np.asarray([np.sin(2 * np.pi * i / 12) for i in np.arange(1, size + 1)])
# cos1 = np.asarray([np.cos(2 * np.pi * i / 12) for i in np.arange(1, size + 1)])
#
# xt = np.cumsum(np.random.normal(scale=0.1, size=size))
#
# yt = xt + beta1 * sin1 + beta2 * cos1 + np.random.normal(scale=0.1, size=size)
#
# series_det = pd.Series(yt)

# decomposition

series_decomp = STL(yt, period=period).fit()

df = pd.DataFrame(
    {
        'Original': yt,
        'Seasonal part': series_decomp.seasonal,
        'Seasonally-adjusted': yt - series_decomp.seasonal,
    }
)

dfr = df.reset_index()
dfm = dfr.melt('Date')

plot = \
    ggplot(dfm) + \
    aes(x='Date', y='value') + \
    facet_grid('variable ~.', scales='free') + \
    theme_538(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.125,
          axis_text=element_text(size=11),
          axis_text_x=element_text(size=9),
          legend_title=element_blank(),
          panel_background=element_rect(fill='white'),
          plot_background=element_rect(fill='white'),
          strip_background=element_rect(fill='white'),
          legend_background=element_rect(fill='white'),
          strip_text=element_text(size=12, color='black'),
          legend_position='none') + \
    labs(x='', y='value') + \
    geom_line(color='#2832c2', size=1)

print(plot)

dfr = df.drop('Seasonal part', axis=1).reset_index()
dfm = dfr.melt('Date')

plot = \
    ggplot(dfm) + \
    aes(x='Date', y='value', color='variable') + \
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
    geom_line(size=1) + \
    scale_color_manual(values=['lightgrey', 'blue'])

# seasonality strength
seasonal_strength(yt, period=12)

# ACF

acorr = AutoCorrelation(n_lags=24, alpha=0.05)

acorr.calc_acf(yt)
acorr.acf_df['Type'] = 'Auto-correlation up to 24 lags'
acorr.acf_df['Laglabel'] = acorr.acf_df['Lag']
acf_series_plot = AutoCorrelation.acf_plot(acorr.acf_df) + facet_grid('~Type')

print(acf_series_plot)

# CH seasonal unit root test
nsdiffs(x=yt, m=period, test='ch')
# nsdiffs(x=yt, m=period, test='ocsb')


##### 7 modeling techniques #######
###################################

# 1. Seasonal dummy variables

from sktime.transformations.series.date import DateTimeFeatures
from sklearn.preprocessing import OneHotEncoder

monthly_feats = DateTimeFeatures(ts_freq='M',
                                 keep_original_columns=False,
                                 feature_scope='efficient')

datetime_feats = monthly_feats.fit_transform(yt)
datetime_feats = datetime_feats.drop('year', axis=1)

encoder = OneHotEncoder(drop='first', sparse=False)
encoded_feats = encoder.fit_transform(datetime_feats)

encoded_feats_df = pd.DataFrame(encoded_feats,
                                columns=encoder.get_feature_names_out(),
                                dtype=int)

# 2. Fourier series

# https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.transformations.series.fourier.FourierFeatures.html

from sktime.transformations.series.fourier import FourierFeatures

fourier = FourierFeatures(sp_list=[12],
                          fourier_terms_list=[4],
                          keep_original_columns=False)

fourier_feats = fourier.fit_transform(yt)

# fourier_feats['sin_12_1'].plot()

dfr = fourier_feats[['sin_12_1','cos_12_1']].reset_index()
dfm = dfr.head(40).melt('Date')

plot = \
    ggplot(dfm) + \
    aes(x='Date', y='value', color='variable') + \
    theme_538(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.15,
          axis_text=element_text(size=12),
          legend_title=element_blank(),
          panel_background=element_rect(fill='white'),
          plot_background=element_rect(fill='white'),
          strip_background=element_rect(fill='white'),
          legend_background=element_rect(fill='white'),
          legend_position='none') + \
    labs(x='', y='') + \
    geom_line(size=1) + \
    scale_color_manual(values=['red', 'blue'])

print(plot)

# 3. Radial Basis Functions
from sklego.preprocessing import RepeatingBasisFunction

rbf_encoder = RepeatingBasisFunction(n_periods=4,
                                     column='month_of_year',
                                     input_range=(1, 12),
                                     remainder='drop',
                                     width=0.25)

rbf_features = rbf_encoder.fit_transform(datetime_feats)
rbf_features_df = pd.DataFrame(rbf_features,
                               columns=[f'RBF{i}'
                                        for i in range(rbf_features.shape[1])])

# rbf_features_df['RBF0'].plot()
dfr = rbf_features_df[['RBF0','RBF1', 'RBF2']].reset_index()
dfm = dfr.head(40).melt('index')

plot = \
    ggplot(dfm) + \
    aes(x='index', y='value', color='variable') + \
    theme_538(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.15,
          axis_text=element_text(size=12),
          legend_title=element_blank(),
          panel_background=element_rect(fill='white'),
          plot_background=element_rect(fill='white'),
          strip_background=element_rect(fill='white'),
          legend_background=element_rect(fill='white'),
          legend_position='none') + \
    labs(x='', y='') + \
    geom_line(size=1) + \
    scale_color_manual(values=['green', 'yellow', 'steelblue'])

print(plot)

# 4. Seasonal Auto-regression
import pmdarima as pm

model = pm.auto_arima(yt, m=12, trace=True)

model.summary()
# Best model:  ARIMA(0,1,0)(1,0,0)[12]

# 5. Adding Extra Variables

temp = pd.read_csv('data/air_temp_final.csv')
temp['datetime'] = \
    pd.to_datetime([f'{year}/{month}/{day} {hour}:00'
                    for year, month, day, hour in zip(temp['calendar_year'],
                                                      temp['month'],
                                                      temp['day'],
                                                      temp['hour'])])

temp = temp.set_index('datetime')['smm1_ta_C']
temp = temp.resample('D').mean()

plot = \
    ggplot(temp.reset_index()) + \
    aes(x='datetime',
        y='smm1_ta_C') + \
    theme_minimal(base_family='Palatino', base_size=12) + \
    theme(plot_margin=0.15,
          axis_text_y=element_text(size=12),
          axis_title_y=element_text(size=12),
          axis_text_x=element_text(angle=0, size=12),
          legend_title=element_blank(),
          panel_background=element_rect(fill='white'),
          plot_background=element_rect(fill='white'),
          strip_background=element_rect(fill='white'),
          legend_background=element_rect(fill='white'),
          legend_position='top') + \
    geom_line(color='#622a0f', size=1) + \
    labs(x='',
         y='Temperature',
         title='')

# 6. Adjustment via Seasonal Differencing

from sklearn.model_selection import train_test_split
from sktime.forecasting.compose import make_reduction
from sklearn.linear_model import RidgeCV

train, test = train_test_split(yt, test_size=12, shuffle=False)

train_sdiff = train.diff(periods=12)[12:]

forecaster = make_reduction(estimator=RidgeCV(),
                            strategy='recursive',
                            window_length=3)

forecaster.fit(train_sdiff)
diff_pred = forecaster.predict(fh=list(range(1, 13)))

# 7. Adjustment via Decomposition

from statsmodels.tsa.api import STL
from sktime.forecasting.naive import NaiveForecaster

series_decomp = STL(yt, period=period).fit()

seas_adj = yt - series_decomp.seasonal

forecaster = make_reduction(estimator=RidgeCV(),
                            strategy='recursive',
                            window_length=3)

forecaster.fit(seas_adj)

seas_adj_pred = forecaster.predict(fh=list(range(1, 13)))

seas_forecaster = NaiveForecaster(strategy='last', sp=12)
seas_forecaster.fit(series_decomp.seasonal)
seas_preds = seas_forecaster.predict(fh=list(range(1, 13)))

preds = seas_adj_pred + seas_preds

# 8. Dynamic Linear Models

##  Refer to link
