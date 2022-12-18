import numpy as np

from pmdarima.datasets import load_airpassengers

from src.heteroskedasticity import Heteroskedasticity
from src.plots.ts_lineplot_simple import LinePlot

series = load_airpassengers(True)
series.name = 'Series'
series.index.name = 'Index'

raw_results = Heteroskedasticity.run_all_tests(series)
log_results = Heteroskedasticity.run_all_tests(np.log(series))

from pprint import pprint

pprint(raw_results)
pprint(log_results)

plot_raw = LinePlot.univariate(series.reset_index(),
                               x_axis_col='Index',
                               y_axis_col='Series',
                               add_smooth=False,
                               x_lab='Index',
                               y_lab='Monthly passengers')

plot_log = LinePlot.univariate(np.log(series).reset_index(),
                               x_axis_col='Index',
                               y_axis_col='Series',
                               add_smooth=False,
                               x_lab='Index',
                               y_lab='Log monthly passengers')

plot_raw.save('raw_passenger_series.pdf', width=8, height=3)
plot_log.save('log_passenger_series.pdf', width=8, height=3)
