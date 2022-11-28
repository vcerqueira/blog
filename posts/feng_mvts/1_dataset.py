import pandas as pd

# skipping second row, setting time column as a datetime column
# dataset available here: https://github.com/vcerqueira/blog/tree/main/data
buoy = pd.read_csv('data/smart_buoy.csv', skiprows=[1], parse_dates=['time'])

# setting time as index
buoy.set_index('time', inplace=True)
# resampling to hourly data
buoy = buoy.resample('H').mean()
buoy.columns = [
    'PeakP', 'PeakD', 'Upcross',
    'SWH', 'SeaTemp', 'Hmax', 'THmax',
    'MCurDir', 'MCurSpd'
]


### plot

import numpy as np
from plotnine import *

buoy_df = pd.melt(np.log(buoy[:'2022-02-01 00:00:00+00:00']).reset_index(), 'time')

plot = \
    ggplot(buoy_df) + \
    aes(x='time',
        y='value',
        group='variable',
        color='variable') + \
    theme_classic(base_family='Palatino', base_size=12) + \
    theme(plot_margin=0.15,
          axis_text_y=element_text(size=10),
          axis_text_x=element_text(angle=30, size=9),
          legend_title=element_blank(),
          legend_position='top')

plot += geom_line()
plot = \
    plot + \
    xlab('') + \
    ylab('Value (log-scale)') + \
    ggtitle('')

plot.save(f'mv_line_plot.pdf', height=5, width=9)
