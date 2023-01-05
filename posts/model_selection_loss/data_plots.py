from plotnine import *
import pandas as pd

COL = '#466dab'

ACCURACY = {
    'Blocked KFold': 10,
    'TimeSeriesSplit': 9,
    'hv-Blocked KFold': 10,
    'Modified-KFold': 7,
    'KFold': 9,
    'TimeSeriesSplit \n (Sliding)': 7,
    'MonteCarloCV': 8,
    'Holdout': 7
}

LOSS_AVG = {
    'Blocked KFold': 0.29,
    'TimeSeriesSplit': .28,
    'hv-Blocked KFold': .3,
    'Modified-KFold': .3,
    'KFold': .32,
    'TimeSeriesSplit \n (Sliding)': .34,
    'MonteCarloCV': .35,
    'Holdout': .58
}

LOSS_SDEV = {
    'Blocked KFold': .68,
    'TimeSeriesSplit': .68,
    'hv-Blocked KFold': .69,
    'Modified-KFold': .66,
    'KFold': .78,
    'TimeSeriesSplit \n (Sliding)': 1.03,
    'MonteCarloCV': .85,
    'Holdout': 1.64
}

df = pd.DataFrame([ACCURACY, LOSS_AVG, LOSS_SDEV]).T
df.reset_index(inplace=True)
df.columns = ['Method', 'Accuracy', 'Average % Diff.', 'Std Loss']
df = df.sort_values('Accuracy', ascending=False)
df['Method'] = pd.Categorical(df['Method'], categories=df['Method'])
df['Upper'] = df['Average % Diff.'] + df['Std Loss']
df['Lower'] = df['Average % Diff.'] - (df['Std Loss'] / 4)

plot_accuracy = ggplot(df, aes(x='Method', y='Accuracy')) + \
                theme_538(base_family='Palatino', base_size=12) + \
                theme(plot_margin=.2,
                      axis_text=element_text(size=12),
                      axis_text_x=element_text(size=10,
                                               angle=60),
                      legend_title=element_blank(),
                      legend_position='top') + \
                geom_bar(position='dodge',
                         stat='identity', fill=COL)

df = df.sort_values('Average % Diff.', ascending=True)
df['Method'] = pd.Categorical(df['Method'], categories=df['Method'].values.to_list())

plot_error = ggplot(df, aes(x='Method', y='Average % Diff.')) + \
             theme_538(base_family='Palatino', base_size=12) + \
             theme(plot_margin=.25,
                   axis_text=element_text(size=10),
                   axis_text_x=element_text(size=10,
                                            angle=60),
                   legend_title=element_blank(),
                   legend_position='top') + \
             geom_errorbar(aes(ymin='Lower', ymax='Upper'),
                           width=.5,
                           position=position_dodge(.9)) + \
             geom_bar(position='dodge',
                      stat='identity', fill=COL)

example_df = [
    ['True', 'True', 'True', 'True',
     'CV1', 'CV1', 'CV1', 'CV1',
     'CV2', 'CV2', 'CV2', 'CV2'],
    ['M1', 'M2', 'M3', 'M4',
     'M1', 'M2', 'M3', 'M4',
     'M1', 'M2', 'M3', 'M4'],
    [1, 2, 3, 4, 2, 1.5, 5, 3,
     1 * 3, 2 * 3, 3 * 3, 4 * 3]
]
example_df = pd.DataFrame(example_df).T
example_df.columns = ['Loss', 'Model', 'value']
example_df['Loss'] = pd.Categorical(example_df['Loss'],
                                    categories=['True', 'CV1', 'CV2'])
example_df['value'] = example_df['value'].astype(float)

example_plot = ggplot(example_df,
                      aes(x='Model', y='value', fill='Loss')) + \
               geom_bar(stat="identity", position=position_dodge()) + \
               theme_538(base_family='Palatino', base_size=12) + \
               theme(plot_margin=.2,
                     axis_text=element_text(size=12),
                     axis_text_x=element_text(size=10),
                     legend_title=element_blank(),
                     legend_position='top') + \
               ylab('Error') + \
               scale_fill_manual(values=['#4682b4', '#a3c585', '#658354'])

plot_accuracy.save('plot_accuracy.pdf', width=10, height=3)
plot_error.save('plot_error.pdf', width=10, height=4)
example_plot.save('example_plot.pdf', width=9, height=4)
