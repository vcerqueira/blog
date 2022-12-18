import numpy as np
from pmdarima.datasets import load_airpassengers

from src.heteroskedasticity import Heteroskedasticity

series = load_airpassengers(True)

test_results = Heteroskedasticity.run_all_tests(series)
log_test_results = Heteroskedasticity.run_all_tests(np.log(series))
