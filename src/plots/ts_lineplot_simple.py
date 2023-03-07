import pandas as pd

from plotnine import *

PISTACHIO = '#58a63e'
PISTACHIO2 = '#b6dea8'


class LinePlot:

    @staticmethod
    def univariate(data: pd.DataFrame,
                   x_axis_col: str,
                   y_axis_col: str,
                   line_color: str = PISTACHIO,
                   x_lab: str = '',
                   y_lab: str = '',
                   title: str = '',
                   add_smooth: bool = False):

        plot = \
            ggplot(data) + \
            aes(x=x_axis_col, y=y_axis_col, group=1) + \
            theme_classic(base_family='Palatino', base_size=12) + \
            theme(plot_margin=.125,
                  axis_text=element_text(size=12),
                  legend_title=element_blank(),
                  legend_position=None)

        if add_smooth:
            plot += geom_smooth(color=PISTACHIO2, size=5)

        plot += geom_line(color=line_color, size=1)

        plot = \
            plot + \
            xlab(x_lab) + \
            ylab(y_lab) + \
            ggtitle(title)

        return plot

    @staticmethod
    def multivariate(data, x, y, group, x_lab, y_lab):
        plot = \
            ggplot(data) + \
            aes(x=x,
                y=y,
                group=group,
                color=group) + \
            theme_538(base_family='Palatino', base_size=12) + \
            theme(plot_margin=.2,
                  axis_text=element_text(size=10),
                  axis_text_x=element_text(size=11),
                  legend_title=element_blank(),
                  legend_position='top')

        plot += geom_line(alpha=0.6, size=1.1)

        plot = \
            plot + \
            xlab(x_lab) + \
            ylab(y_lab) + \
            ggtitle('')

        return plot
