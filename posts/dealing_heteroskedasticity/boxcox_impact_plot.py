from plotnine import *
import pandas as pd
from pmdarima.datasets import load_airpassengers
from scipy.stats import boxcox

series = load_airpassengers(True)

series_transformed, lambda_ = boxcox(series)

df = pd.concat([series, pd.Series(series_transformed)], axis=1)
df.reset_index(inplace=True)
df.columns = ['Index', 'Original', 'Transformed']

df_melted = pd.melt(df, 'Index')

plot = \
    ggplot(df_melted) + \
    aes(x='Index', y='value') + \
    facet_grid('variable ~.', scales='free') + \
    theme_538(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.125,
          axis_text=element_text(size=11),
          legend_title=element_blank(),
          strip_text=element_text(size=13),
          legend_position='none')

plot += geom_line(color='#58a63e', size=1)

plot = \
    plot + \
    xlab('') + \
    ylab('') + \
    ggtitle('')

plot.save('log_impact_grid.pdf', height=4, width=7)
