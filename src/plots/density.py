import numpy as np
import pandas as pd

from plotnine import *

PISTACHIO = '#58a63e'
BROWN = '#cf8806'


class PlotDensity:

    @staticmethod
    def by_pair(data: pd.DataFrame,
                x_axis_col: str,
                group_col: str,
                x_lab: str = '',
                y_lab: str = '',
                title: str = ''):
        COLORS = [PISTACHIO, BROWN]

        data_grp = data.groupby(group_col).mean().reset_index()
        data_grp.set_index(group_col, inplace=True)

        plot = ggplot(data) + \
               aes(x=x_axis_col, color=group_col, fill=group_col) + \
               theme_538(base_family='Palatino', base_size=12) + \
               theme(plot_margin=.175,
                     axis_text=element_text(size=12),
                     strip_text=element_text(size=12),
                     legend_title=element_blank(),
                     legend_position='top')

        for i, grp in enumerate(data_grp[x_axis_col]):
            plot += geom_vline(xintercept=grp,
                               linetype='dashed',
                               color=COLORS[i],
                               size=1.1,
                               alpha=0.7)

        plot = plot + \
               geom_density(aes(y=after_stat('density')), alpha=.3) + \
               xlab(x_lab) + \
               ylab(y_lab) + \
               ggtitle(title) + \
               scale_fill_manual(values=COLORS) + \
               scale_color_manual(values=COLORS)

        return plot

    @staticmethod
    def univariate_plus_hlines(data: pd.DataFrame,
                               hline_values: np.ndarray,
                               x_axis_col: str,
                               x_lab: str = '',
                               y_lab: str = '',
                               title: str = ''):

        plot = ggplot(data) + \
               aes(x=x_axis_col) + \
               theme_538(base_family='Palatino', base_size=12) + \
               theme(plot_margin=.175,
                     axis_text=element_text(size=12),
                     strip_text=element_text(size=12),
                     legend_title=element_blank(),
                     legend_position='top')

        for i, val in enumerate(hline_values):
            plot += geom_vline(xintercept=val,
                               linetype='dashed',
                               color=BROWN,
                               size=1.1,
                               alpha=0.7)

        plot = plot + \
               geom_density(aes(y=after_stat('density')),
                            alpha=.3,
                            color=PISTACHIO,
                            fill=PISTACHIO) + \
               xlab(x_lab) + \
               ylab(y_lab) + \
               ggtitle(title)

        return plot
