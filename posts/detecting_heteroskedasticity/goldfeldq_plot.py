from plotnine import *
import pandas as pd

from pmdarima.datasets import load_airpassengers

from src.heteroskedasticity import Heteroskedasticity
from src.plots.variance_dist import VarianceDistPlot

series = load_airpassengers(True)

residuals = Heteroskedasticity.get_residuals(series)

partition_size = 0.5

residuals_df = residuals.reset_index()
residuals_df.columns = ['Time', 'Residuals']

n = residuals.shape[0]

p1 = residuals.head(int(n * partition_size))
p2 = residuals.tail(int(n * partition_size))

p1_df = pd.DataFrame({'Residuals': p1, 'Id': range(len(p1))})
p1_df['Part'] = 'First Half \nof Residuals'

p2_df = pd.DataFrame({'Residuals': p2, 'Id': range(len(p2))})
p2_df['Part'] = 'Second Half \nof Residuals'

df = pd.concat([p1_df, p2_df])

plot = VarianceDistPlot.plot(data=df,
                             x_axis_col='Part',
                             y_axis_col='Residuals',
                             group_col='Part')

plot.save('heteroskedasticity.pdf', width=8, height=4)
