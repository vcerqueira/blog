import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import plotnine as p9

# RANDOM WALKS

rw = pd.Series(np.cumsum(np.random.choice([-1, 1], size=1000)))
rw_df = rw.reset_index()
rw_df.columns = ['index', 'value']
rw_df['moving_average'] = rw_df['value'].rolling(20).mean()

rw_plot = \
    p9.ggplot(data=rw_df, mapping=p9.aes(x='index', y='value')) + \
    p9.theme_classic(base_family='Palatino', base_size=12) + \
    p9.theme(plot_margin=.175,
             panel_background=p9.element_rect(fill='white'),
             plot_background=p9.element_rect(fill='white'),
             strip_background=p9.element_rect(fill='white'),
             legend_background=p9.element_rect(fill='white'),
             legend_title=p9.element_blank(),
             legend_position='none') + \
    p9.geom_line(color="#e1c564", size=1) + \
    p9.labs(x='', y='')

print(rw_plot)

# moving average filter

rw_plot = \
    p9.ggplot(data=rw_df, mapping=p9.aes(x='index', y='moving_average')) + \
    p9.theme_classic(base_family='Palatino', base_size=12) + \
    p9.theme(plot_margin=.175,
             panel_background=p9.element_rect(fill='white'),
             plot_background=p9.element_rect(fill='white'),
             strip_background=p9.element_rect(fill='white'),
             legend_background=p9.element_rect(fill='white'),
             legend_title=p9.element_blank(),
             legend_position='none') + \
    p9.geom_line(color="#e1c564", size=1) + \
    p9.labs(x='', y='')

print(rw_plot)

# MA(1)

import statsmodels.api as sm

ma1 = sm.tsa.arma_generate_sample(ar=[1], ma=[1, 0.3], nsample=1000)
ma3 = sm.tsa.arma_generate_sample(ar=[1], ma=[1, 0.7, 0.3, 0.55], nsample=1000)

rw_df['ma1'] = ma1
rw_df['ma3'] = ma3

ma1_plot = \
    p9.ggplot(data=rw_df.head(300), mapping=p9.aes(x='index', y='ma1')) + \
    p9.theme_classic(base_family='Palatino', base_size=12) + \
    p9.theme(plot_margin=.175,
             panel_background=p9.element_rect(fill='white'),
             plot_background=p9.element_rect(fill='white'),
             strip_background=p9.element_rect(fill='white'),
             legend_background=p9.element_rect(fill='white'),
             legend_title=p9.element_blank(),
             legend_position='none') + \
    p9.geom_line(color="#e1c564", size=1) + \
    p9.labs(x='', y='')

print(ma1_plot)

ma3_plot = \
    p9.ggplot(data=rw_df.head(300), mapping=p9.aes(x='index', y='ma3')) + \
    p9.theme_classic(base_family='Palatino', base_size=12) + \
    p9.theme(plot_margin=.175,
             panel_background=p9.element_rect(fill='white'),
             plot_background=p9.element_rect(fill='white'),
             strip_background=p9.element_rect(fill='white'),
             legend_background=p9.element_rect(fill='white'),
             legend_title=p9.element_blank(),
             legend_position='none') + \
    p9.geom_line(color="#e1c564", size=1) + \
    p9.labs(x='', y='')

print(ma3_plot)

from statsforecast.models import ARIMA

ma1 = ARIMA(order=(0, 0, 1), season_length=1)

ma1 = ma1.fit(y=np.random.random(100))
ma1.predict(h=10)

