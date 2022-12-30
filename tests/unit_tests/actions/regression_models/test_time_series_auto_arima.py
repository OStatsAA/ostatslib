# pylint: disable=redefined-outer-name
"""
time_series_auto_arima action testing module
"""

from pandas import DataFrame, date_range
from scipy.stats import norm
import numpy as np
import pytest

from ostatslib.actions import time_series_auto_arima
from ostatslib.states import State


@pytest.fixture
def dummy_ar_1_dataset() -> DataFrame:
    """
    Generates a dummy AR(1) dataset
    """
    dates = date_range(start='2017/1/1', end='2023/7/15', freq='D')
    data_length = len(dates)
    results = np.ones(data_length)
    data = DataFrame({'time': dates.to_list(), 'result': results})

    for i in range(2, data_length):
        data.loc[i, 'result'] += 0.8 * \
            data.loc[i-1, 'result'] + norm.rvs(scale=0.5)

    return data


@pytest.fixture
def dummy_one_variable_dataset() -> DataFrame:
    """
    Generates a dummy AR(1) dataset
    """
    dates = date_range(start='2017/1/1', end='2023/7/15', freq='D')
    data_length = len(dates)
    results = np.ones(data_length)
    x1 = np.random.poisson(2, data_length)
    data = DataFrame({'time': dates.to_list(), 'x1': x1, 'result': results})

    for i in range(2, data_length):
        data.loc[i, 'result'] += 0.8 * data.loc[i-1, 'result'] + \
            data.loc[i, 'x1'] + norm.rvs(scale=0.5)

    return data


def test_if_state_is_valid(dummy_ar_1_dataset) -> None:
    """
    Tests if action yields negative reward if state is invalid
    """
    state = State()
    result = time_series_auto_arima(state, dummy_ar_1_dataset)
    assert result.reward < 0


def test_simple_time_series(dummy_ar_1_dataset) -> None:
    """
    Tests simple ar1
    """
    state = State()
    state.set("time_convertable_variable", "time")
    state.set("is_response_quantitative", 1)
    result = time_series_auto_arima(state, dummy_ar_1_dataset)
    assert result.reward > 0


def test_time_series_with_1_variable(dummy_one_variable_dataset) -> None:
    """
    Tests ar1 with one variable
    """
    state = State()
    state.set("time_convertable_variable", "time")
    state.set("is_response_quantitative", 1)
    result = time_series_auto_arima(state, dummy_one_variable_dataset)
    assert result.reward > 0
