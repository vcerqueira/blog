import pandas as pd

from plotnine import *

wine = pd.read_csv('data/wine_sales.csv', parse_dates=['date'])

# setting date as index
wine.set_index('date', inplace=True)

wine_df = pd.melt(wine.reset_index(), 'date')
# wine_df = pd.melt((wine / wine.std()).reset_index(), 'date')

plot = \
    ggplot(wine_df) + \
    aes(x='date',
        y='value',
        group='variable',
        color='variable') + \
    theme_538(base_family='Palatino', base_size=12) + \
    theme(plot_margin=0.2,
          axis_text=element_text(size=8),
          axis_text_x=element_text(angle=30),
          legend_title=element_blank(),
          legend_position='top')

plot += geom_line()
plot = \
    plot + \
    xlab('') + \
    ylab('Wine Sales') + \
    ggtitle('')

PISTACHIO_BLACK = '#2b5c0e'
PISTACHIO_FILL = '#edf7ea'

plot += geom_vline(xintercept=wine_df['date'][897],
                   linetype='dashed',
                   color=PISTACHIO_BLACK,
                   size=1.1)

plot += geom_label(label='Testing Start',
                   # y=int(data['Series'].max() * .95),
                   y=int(wine_df['value'].max() * .9),
                   fill=PISTACHIO_FILL,
                   color=PISTACHIO_BLACK,
                   x=wine_df['date'][897],
                   size=10)

plot.save(f'mv_line_plot.pdf', height=4, width=8)
