import pandas as pd
from plotnine import *


def cv_plot(cv, X, y):
    segments_data = []
    for i, (tr, ts) in enumerate(cv.split(X, y)):
        segments_data.append({'y': i + 1, 'x_min': tr[0], 'x_max': tr[-1], 'part': 'Training'})
        segments_data.append({'y': i + 1, 'x_min': ts[0], 'x_max': ts[-1], 'part': 'Validation'})

    segments_df = pd.DataFrame(segments_data)

    plot = \
        ggplot(segments_df) + \
        aes(x='x_min', xend='x_max', y='y', yend='y', color='part') + \
        theme_minimal(base_family='Palatino', base_size=10) + \
        theme(plot_margin=.25,
              axis_text=element_text(size=8),
              legend_title=element_blank(),
              legend_position='right') + \
        geom_segment(
            size=3,
        ) + \
        xlab('Data Index') + \
        ylab('CV Iteration') + \
        ggtitle(f'Data Partitions with {cv.__class__.__name__}') + \
        scale_y_reverse() + \
        scale_color_manual(values=['#58a63e', '#cf8806'])

    return plot
