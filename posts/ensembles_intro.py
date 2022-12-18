import pandas as pd
from statsforecast.models import ETS
from statsforecast.models import AutoARIMA
from statsforecast.models import SimpleExponentialSmoothingOptimized
from statsforecast.models import SeasonalNaive
from statsforecast.models import AutoARIMA, SimpleExponentialSmoothingOptimized, SeasonalNaive
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sktime.forecasting.theta import ThetaForecaster

from src.metrics import mape
from config.datasets import TOURISM, DATASETS_DIR, EXPERIMENTS_DIR, RESULTS_DIR

PROBLEM = 'tourism'
METHOD = 'ets'
METHOD_SHORT = 'ETS'
fh = TOURISM['horizons']

train_yearly_path = f'{DATASETS_DIR}/{PROBLEM}/{TOURISM["train"]["yearly"]}'
train_quarterly_path = f'{DATASETS_DIR}/{PROBLEM}/{TOURISM["train"]["quarterly"]}'
train_monthly_path = f'{DATASETS_DIR}/{PROBLEM}/{TOURISM["train"]["monthly"]}'
test_yearly_path = f'{DATASETS_DIR}/{PROBLEM}/{TOURISM["test"]["yearly"]}'
test_quarterly_path = f'{DATASETS_DIR}/{PROBLEM}/{TOURISM["test"]["quarterly"]}'
test_monthly_path = f'{DATASETS_DIR}/{PROBLEM}/{TOURISM["test"]["monthly"]}'

experiments_path = f'{EXPERIMENTS_DIR}/{PROBLEM}/{METHOD}'

train_yearly = pd.read_csv(train_yearly_path)
train_monthly = pd.read_csv(train_monthly_path)
train_quarterly = pd.read_csv(train_quarterly_path)
test_yearly = pd.read_csv(test_yearly_path)
test_monthly = pd.read_csv(test_monthly_path)
test_quarterly = pd.read_csv(test_quarterly_path)

y_ids = train_yearly.columns.tolist()
m_ids = train_monthly.columns.tolist()
q_ids = train_quarterly.columns.tolist()

series_ids = y_ids + m_ids + q_ids

mape_values = {'st': {}, 'mt': {}, 'lt': {}}
for id in series_ids:
    # id = 'Y1'
    print(id)

    if id.startswith('Y'):
        train_series = train_yearly[id][2:].dropna().values
        test_series = test_yearly[id][2:].dropna().values
        h_name = 'yearly'
    elif id.startswith('m'):
        train_series = train_monthly[id][3:].dropna().values
        test_series = test_monthly[id][3:].dropna().values
        h_name = 'monthly'
    else:
        train_series = train_quarterly[id][3:].dropna().values
        test_series = test_quarterly[id][3:].dropna().values
        h_name = 'quarterly'

    freq = TOURISM['frequency'][h_name]
    h = TOURISM['h'][h_name]

    model = ETS(season_length=freq)
    model.fit(train_series)
    forecasts = model.forecast(y=train_series, h=h)['mean']

    model = AutoARIMA(season_length=freq, approximation=True)
    model.fit(train_series)
    forecasts = model.forecast(y=train_series, h=h)['mean']

    model = SimpleExponentialSmoothingOptimized()
    model.fit(train_series)
    forecasts = model.forecast(y=train_series, h=h)['mean']




    model = SeasonalNaive(season_length=freq)
    model.fit(train_series)
    forecasts = model.forecast(y=train_series, h=h)['mean']

    mape_vals = mape(forecasts, test_series)

    for h_vb, h in fh[h_name].items():
        avg_mape = mape_vals[:h].mean()
        mape_values[h_vb][id] = avg_mape

for h in mape_values:
    print(h)
    model_mape = pd.Series(mape_values[h]).reset_index()
    model_mape.columns = ['ID', METHOD_SHORT]
    model_mape.to_csv(f'{RESULTS_DIR}/{PROBLEM}/models/{h}/{METHOD_SHORT}.csv', index=False)
