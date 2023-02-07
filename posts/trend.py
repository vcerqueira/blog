import pandas as pd
from plotnine import *

from pmdarima.datasets import load_airpassengers

from src.plots.ts_lineplot_simple import LinePlot

series = load_airpassengers(True)
series.index = pd.date_range(start=pd.Timestamp('1949-01-01'), periods=len(series), freq='MS')

series_df = series.reset_index()
series_df.columns = ['Date', 'value']
series_df['Type'] = 'Time series plot with overlayed trend'

plot = LinePlot.univariate(series_df,
                           x_axis_col='Date',
                           y_axis_col='value',
                           line_color='#0058ab',
                           y_lab='No. of Passengers',
                           add_smooth=True)
plot += facet_wrap('~ Type',nrow=1)
plot += theme(strip_text=element_text(size=14))
print(plot)
