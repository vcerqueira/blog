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

dataset, *_ = M4.load('./data', 'Monthly')

# Examples

# permanent level shift
series_pls = dataset.query(f'unique_id=="M1430"').reset_index(drop=True)
series_pls['time'] = pd.date_range(end='2023-12-01', periods=series_pls.shape[0], freq='M')

plot1 = p9.ggplot(data=series_pls) + \
        p9.aes(x='time', y='y') + \
        MY_THEME + \
        p9.geom_line(size=1) + \
        p9.labs(x='', y='')

# transitory level shift
series_tls = dataset.query(f'unique_id=="M1310"').reset_index(drop=True)
series_tls['time'] = pd.date_range(end='2023-12-01', periods=series_tls.shape[0], freq='M')

plot2 = p9.ggplot(data=series_tls) + \
        p9.aes(x='time', y='y') + \
        MY_THEME + \
        p9.geom_line(size=1) + \
        p9.labs(x='', y='')

# permanent variance change
series_pvar = dataset.query(f'unique_id=="M9"').reset_index(drop=True)
series_pvar['time'] = pd.date_range(end='2023-12-01', periods=series_pvar.shape[0], freq='M')

plot3 = p9.ggplot(data=series_pvar) + \
        p9.aes(x='time', y='y') + \
        MY_THEME + \
        p9.geom_line(size=1) + \
        p9.labs(x='', y='')

# vol clustering
buoy = pd.read_csv('data/smart_buoy.csv', skiprows=[1], parse_dates=['time'])
buoy.set_index('time', inplace=True)
buoy = buoy.resample('H').mean()
wave_series = buoy['SignificantWaveHeight']
series_wave_diff = wave_series.diff().head(1500).reset_index()

plot4 = p9.ggplot(data=series_wave_diff) + \
        p9.aes(x='time', y='SignificantWaveHeight') + \
        MY_THEME + \
        p9.geom_line(size=1) + \
        p9.labs(x='', y='')

# Change point detection

from kats.tests.detectors.test_cusum_detection import CUSUMDetector
from kats.consts import TimeSeriesData

series = dataset.query(f'unique_id=="M1430"').reset_index(drop=True)
series['time'] = pd.date_range(end='2023-12-01', periods=series.shape[0], freq='M')
series = series[['time', 'y']]

ts = TimeSeriesData(df=series)

model = CUSUMDetector(ts)
change_points = model.detector(direction=['decrease', 'increase'])
model.plot(change_points)

#

from kats.tests.detectors.test_robust_stat_detection import RobustStatDetector

model = RobustStatDetector(ts)
change_points = model.detector(p_value_cutoff=0.001, comparison_window=12)
model.plot(change_points)

# Ruptures

series['y'].plot()

import matplotlib.pyplot as plt
import ruptures as rpt

signal = series['y'].values

model = rpt.Pelt(model="rbf").fit(signal)
result = model.predict(pen=10)

rpt.display(signal, result, result)
plt.show()

#
series = dataset.query(f'unique_id=="M1430"').reset_index(drop=True)
series['time'] = pd.date_range(end='2023-12-01', periods=series.shape[0], freq='M')
series = series[['time', 'y']]
series['diffed'] = series['y'].diff()

plot = p9.ggplot(data=series) + \
       p9.aes(x='time', y='diffed') + \
       MY_THEME + \
       p9.geom_line(size=1) + \
       p9.labs(x='', y='')

series['step'] = pd.Series(series.index > 190).astype(int)

plot = p9.ggplot(data=series) + \
       p9.aes(x='time', y='step') + \
       MY_THEME + \
       p9.geom_line(size=1) + \
       p9.labs(x='', y='')
