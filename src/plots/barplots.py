import pandas as pd
from plotnine import *


def err_barplot(err: pd.Series):
    err_df = err.reset_index()
    err_df.columns = ['Model', 'Error']
    err_df = err_df.sort_values('Error')
    err_df['Model'] = pd.Categorical(err_df['Model'], categories=err_df['Model'])

    plot = ggplot(err_df, aes(x='Model', y='Error')) + \
           theme_538(base_family='Palatino', base_size=12) + \
           theme(plot_margin=.2,
                 axis_text=element_text(size=12),
                 axis_text_x=element_text(size=10, angle=30),
                 legend_title=element_blank(),
                 legend_position='top') + \
           geom_bar(position='dodge',
                    stat='identity',
                    fill='#466dab')

    plot = \
        plot + \
        xlab('') + \
        ylab('Error') + \
        ggtitle('')

    return plot
