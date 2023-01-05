import pandas as pd


def as_categorical(data: pd.DataFrame, variable: str):
    assert variable in data.columns, 'column unkn'

    unq_values = data[variable].unique()

    var_as_cat = pd.Categorical(data[variable], categories=unq_values)

    return var_as_cat
