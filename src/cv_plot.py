import pandas as pd
import numpy as np
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


def cv_plot_points(cv, X, y):
    segments_data = []
    for i, (tr, ts) in enumerate(cv.split(X, y)):
        tr_df = pd.DataFrame(tr)
        tr_df.columns = ['x']
        tr_df['part'] = 'Training'
        ts_df = pd.DataFrame(ts)
        ts_df.columns = ['x']
        ts_df['part'] = 'Validation'

        df = pd.concat([tr_df, ts_df], axis=0)
        df['size'] = 1.5
        unused = np.setdiff1d(np.arange(X.shape[0]), df['x'].values)

        unused_df = pd.DataFrame(unused)
        unused_df.columns = ['x']
        unused_df['part'] = 'Unused'
        unused_df['size'] = 1
        df = pd.concat([df, unused_df], axis=0)

        df['y'] = i

        segments_data.append(df)

    segments_df = pd.concat(segments_data, axis=0)
    segments_df['part'] = pd.Categorical(segments_df['part'], ['Unused', 'Training', 'Validation'])
    segments_df = segments_df.sort_values('part')

    segments_df['y'] += 1

    plot = \
        ggplot(segments_df) + \
        aes(x='x', y='y', color='part', size='size') + \
        theme_minimal(base_family='Palatino', base_size=10) + \
        theme(plot_margin=.25,
              axis_text=element_text(size=8),
              legend_title=element_blank(),
              legend_position='right') + \
        geom_point(shape="s") + \
        xlab('Data Index') + \
        ylab('CV Iteration') + \
        ggtitle(f'Data Partitions with {cv.__class__.__name__}') + \
        scale_y_reverse() + \
        scale_color_manual(values=['black', '#58a63e', '#cf8806']) + \
        guides(size=None)

    return plot, segments_df
