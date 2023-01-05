import pandas as pd

from plotnine import *
from numerize import numerize

from src.utils.categories import as_categorical


PISTACHIO_HARD = '#58a63e'
PISTACHIO_BLACK = '#2b5c0e'
PISTACHIO_MID = '#a9d39e'


class SummaryStatPlot:
    """
    Mean plot & Std plot
    """

    @staticmethod
    def summary_by_group(
            data: pd.DataFrame,
            y_col: str,
            group_col: str,
            func: str):
        assert func in ['mean', 'median', 'std', 'var']

        grouped_df = data.groupby(group_col)[y_col]
        target_series = data[y_col]

        group_stat = getattr(grouped_df, func)()
        overall_stat = getattr(target_series, func)()

        return group_stat, overall_stat

    @classmethod
    def SummaryPlot(cls,
                    data: pd.DataFrame,
                    y_col: str,
                    group_col: str,
                    func: str,
                    numerize_y: bool = True,
                    x_lab: str = '',
                    y_lab: str = '',
                    title: str = ''):
        """
        todo Anova/kruskal post hoc comparison to check which groups differ
        todo Posso decidir mostrar a meanplot ou stdplot se uma moving average ajudar nas previsões?

        """

        group_stat, overall_stat = cls.summary_by_group(data=data,
                                                        y_col=y_col,
                                                        group_col=group_col,
                                                        func=func)

        group_stat_df = group_stat.reset_index()
        group_stat_df[group_col] = as_categorical(group_stat_df, group_col)

        plot = \
            ggplot(group_stat_df) + \
            aes(x=group_col, y=y_col, group=1) + \
            theme_538(base_family='Palatino', base_size=12) + \
            theme(plot_margin=.125,
                  axis_text=element_text(size=10),
                  legend_title=element_blank(),
                  legend_position=None)

        plot += geom_line(color=PISTACHIO_MID, size=.7, linetype='dashed')
        plot += geom_point(color=PISTACHIO_HARD, size=4)
        plot += geom_hline(yintercept=overall_stat,
                           linetype='solid',
                           color=PISTACHIO_BLACK,
                           size=1.1)

        if numerize_y:
            plot = \
                plot + \
                scale_y_continuous(labels=lambda lst: [numerize.numerize(x) for x in lst])

        plot = \
            plot + \
            xlab(x_lab) + \
            ylab(y_lab) + \
            ggtitle(title)

        return plot
