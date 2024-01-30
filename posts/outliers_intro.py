from datasetsforecast.m4 import M4
import plotnine as p9
import pandas as pd
import numpy as np

MY_THEME = p9.theme_538(base_family='Palatino', base_size=12) + \
           p9.theme(plot_margin=.125,
                    axis_text_y=p9.element_text(size=10),
                    panel_background=p9.element_rect(fill='white'),
                    plot_background=p9.element_rect(fill='white'),
                    strip_background=p9.element_rect(fill='white'),
                    legend_background=p9.element_rect(fill='white'),
                    axis_text_x=p9.element_text(size=10))

# Example series and plotting outliers

dataset, *_ = M4.load('./data', 'Hourly')

series = dataset.query(f'unique_id=="H1"').reset_index(drop=True)

series['diff_y'] = series['y'].diff().diff()
series['diff_y'] += np.abs(series['diff_y'].min())
series['is outlier'] = (series['diff_y'] > 100).astype(int)
series['is innov outlier'] = series['is outlier'].copy()
series['diff_y_innov'] = series['diff_y'].copy()

outlier_idx = np.where(series['is outlier'] > 0)[0]

for i, idx in enumerate(outlier_idx):
    for j in range(1, 11):
        series['diff_y_innov'][j + idx] += 50
        series['is innov outlier'][j + idx] = 1

    for j in range(10, 21):
        series['diff_y_innov'][j + idx] += 20
        series['is innov outlier'][j + idx] = 1

innov_outlier_idx = np.where(series['is innov outlier'] > 0)[0]
# series['diff_y'].plot()
# series['diff_y_innov'].plot()
# series['col1'] = ['steelblue' if x > 0 else 'red' for x in series['is outlier']]

series['index'] = pd.date_range(end='2023-12-01', periods=series.shape[0], freq='H')

time_plot = p9.ggplot(data=series) + \
            p9.aes(x='index', y='diff_y') + \
            MY_THEME + \
            p9.geom_line(size=1) + \
            p9.labs(x='', y='')

for id_ in series['index'][outlier_idx]:
    time_plot += p9.geom_vline(xintercept=id_,
                               linetype='solid',
                               color='orange',
                               alpha=0.5,
                               size=2)

time_plot2 = p9.ggplot(data=series) + \
             p9.aes(x='index', y='diff_y_innov') + \
             MY_THEME + \
             p9.geom_line(size=1) + \
             p9.labs(x='', y='')

for id1_ in series['index'][outlier_idx]:
    time_plot2 += p9.geom_vline(xintercept=id1_,
                                linetype='solid',
                                color='orange',
                                alpha=0.6,
                                size=2)

for id2_ in series['index'][innov_outlier_idx]:
    time_plot2 += p9.geom_vline(xintercept=id2_,
                                linetype='solid',
                                color='orange',
                                alpha=0.1,
                                size=2)

# identifying outliers

## prediction-based models

from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

series = dataset.query(f'unique_id=="H1"').reset_index(drop=True)
# series['y'] = series['y'].diff().diff()
series = series.dropna()

model = [SeasonalNaive(season_length=24)]

sf = StatsForecast(df=series, models=model, freq='H')
sf.forecast(h=1, level=[99], fitted=True)

preds = sf.forecast_fitted_values()

is_outlier = (preds['y'] >= preds['SeasonalNaive-hi-99']) | (preds['y'] <= preds['SeasonalNaive-lo-99'])
outliers = preds.loc[is_outlier]

StatsForecast.plot(preds, plot_random=False, plot_anomalies=True)

sn_outlier_idx = np.where(is_outlier)[0]

series['index'] = pd.date_range(end='2023-12-01', periods=series.shape[0], freq='H')
series = series.reset_index(drop=True)

time_plot3 = p9.ggplot(data=series) + \
            p9.aes(x='index', y='y') + \
            MY_THEME + \
            p9.geom_line(size=1) + \
            p9.labs(x='', y='')

for id_ in series['index'][sn_outlier_idx]:
    time_plot3 += p9.geom_vline(xintercept=id_,
                               linetype='solid',
                               color='orange',
                               alpha=0.4,
                               size=2)

## zscore

# values above/below 3 std deviations
thresh = 3

rolling_series = series['y'].rolling(window=24, min_periods=1, center=True)
avg = rolling_series.mean()
std = rolling_series.std(ddof=0)
zscore = series['y'].sub(avg).div(std)
m = zscore.between(-thresh, thresh)

## residuals

from statsmodels.tsa.seasonal import STL

stl = STL(series['y'].values, period=24, robust=True).fit()
resid = pd.Series(stl.resid)

q1, q3 = resid.quantile([.25, .75])
iqr = q3 - q1

is_outlier_r = ~resid.apply(lambda x: q1 - (3 * iqr) < x < q3 + (3 * iqr))
is_outlier_r_idx = np.where(is_outlier_r)[0]

resid_df = resid.reset_index()
resid_df['index'] = pd.date_range(end='2021-12-01', periods=series.shape[0], freq='H')
resid_df.columns = ['index', 'Residual']

time_plot3 = p9.ggplot(data=resid_df) + \
             p9.aes(x='index', y='Residual') + \
             MY_THEME + \
             p9.geom_line(size=1) + \
             p9.labs(x='', y='') + p9.theme(axis_text=p9.element_blank())

for id_ in resid_df['index'][is_outlier_r_idx]:
    time_plot3 += p9.geom_vline(xintercept=id_,
                                linetype='solid',
                                color='orange',
                                alpha=0.3,
                                size=1.2)


#

series['y'][is_outlier_r] = np.nan

series['y'].interpolate()


