import pandas as pd
from plotnine import *


def train_test_yhat_plot(train: pd.Series, test: pd.Series, yhat: pd.DataFrame):
    """

    :param train: time series training data as pd.Series
    :param test: time series testing data as pd.Series
    :param yhat: forecasts as a pd.DF, with the same index as test. a column for each model
    """

    yhat.index = test.index
    yhat.index.name = 'Date'

    yhat = yhat.reset_index().melt('Date')
    #yhat['size'] = .3

    train_df_p = train.reset_index()
    train_df_p.columns = ['Date', 'value']
    train_df_p['variable'] = 'Train'
    #train_df_p['size'] = .3

    test_df_p = test.reset_index()
    test_df_p.columns = ['Date', 'value']
    test_df_p['variable'] = 'Test'
    #test_df_p['size'] = .3

    df = pd.concat([train_df_p, test_df_p, yhat], axis=0)
    df['variable'] = pd.Categorical(df['variable'], categories=df['variable'].unique())

    plot = \
        ggplot(df) + \
        aes(x='Date',
            y='value',
            group='variable',
            color='variable') + \
        theme_538(base_family='Palatino', base_size=12) + \
        theme(plot_margin=.2,
              axis_text=element_text(size=10),
              axis_text_x=element_text(),
              legend_title=element_blank(),
              legend_position='top') #+ guides(size=False)

    plot += geom_line(size=1)

    plot = \
        plot + \
        xlab('') + \
        ylab('') + \
        ggtitle('')

    return plot
