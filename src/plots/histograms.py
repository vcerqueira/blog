import pandas as pd

from plotnine import *
from numerize import numerize

PISTACHIO_HARD = '#58a63e'
PISTACHIO_SOFT = '#b6dea8'


class PlotHistogram:

    @staticmethod
    def univariate(data: pd.DataFrame,
                   x_axis_col: str,
                   n_bins: int,
                   numerize_x: bool = True,
                   x_lab: str = '',
                   y_lab: str = '',
                   title: str = ''):
        """


        :param numerize_x:
        :param data:
        :param x_axis_col:
        :param n_bins:
        :param x_lab:
        :param y_lab:
        :param title:
        :return:
        """
        plot = ggplot(data) + \
               aes(x=x_axis_col) + \
               theme_538(base_family='Palatino', base_size=12) + \
               theme(plot_margin=.15,
                     axis_text=element_text(size=12),
                     # panel_background=element_rect(fill=WHITE),
                     # plot_background=element_rect(fill=WHITE),
                     # strip_background=element_rect(fill=WHITE),
                     # legend_background=element_rect(fill=WHITE),
                     legend_title=element_blank(),
                     legend_position=None) + \
               geom_histogram(alpha=.95,
                              bins=n_bins,
                              color=PISTACHIO_SOFT,
                              fill=PISTACHIO_HARD)

        plot = \
            plot + \
            xlab(x_lab) + \
            ylab(y_lab) + \
            ggtitle(title)

        if numerize_x:
            plot = \
                plot + \
                scale_x_continuous(labels=lambda lst: [numerize.numerize(x) for x in lst]) + \
                scale_y_continuous(labels=lambda lst: [numerize.numerize(x) for x in lst])

        return plot

    @staticmethod
    def by_group(data: pd.DataFrame,
                 x_axis_col: str,
                 group_col: str,
                 x_lab: str = '',
                 y_lab: str = '',
                 title: str = ''):
        plot = ggplot(data) + \
               aes(x=x_axis_col, fill=group_col) + \
               theme_minimal(base_family='Palatino', base_size=12) + \
               theme(plot_margin=.15,
                     axis_text=element_text(size=12),
                     # panel_background=element_rect(fill=WHITE),
                     # plot_background=element_rect(fill=WHITE),
                     # strip_background=element_rect(fill=WHITE),
                     # legend_background=element_rect(fill=WHITE),
                     legend_title=element_blank(),
                     legend_position=None) + \
               geom_histogram(alpha=.9,
                              bins=50)

        plot = \
            plot + \
            xlab(x_lab) + \
            ylab(y_lab) + \
            ggtitle(title)

        return plot
