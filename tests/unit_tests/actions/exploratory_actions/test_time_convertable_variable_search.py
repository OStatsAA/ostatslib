# pylint: disable=redefined-outer-name
"""
test_time_convertible_variable_search action testing module
"""

from pandas import DataFrame, date_range, timedelta_range
import numpy as np
import pytest

from ostatslib.actions import time_convertible_variable_search
from ostatslib.states import State


@pytest.fixture
def dummy_time_series_data() -> DataFrame:
    """
    Generates a dummy dataset base on a time series
    """
    dates = date_range(start='2017/1/1', end='2023/7/15', freq='D')
    results = np.random.randint(0, 42, size=(len(dates)))
    return DataFrame({'time': dates.to_list(), 'result': results})


def test_finding_one_time_convertible_variable(dummy_time_series_data) -> None:
    """
    Tests if action can find date variable when there's only one convertable
    """
    state = State()
    state, reward, _ = time_convertible_variable_search(state, dummy_time_series_data)
    assert bool(state.get('time_convertible_variable')) and reward > 0


def test_finding_two_time_convertible_variable(dummy_time_series_data) -> None:
    """
    Tests if action can find date variable when there's multiple convertable variables
    """
    state = State()
    dummy_time_series_data['time_deltas'] = timedelta_range(start='1 day',
                                                            periods=len(dummy_time_series_data))
    state, reward, _ = time_convertible_variable_search(state, dummy_time_series_data)
    assert bool(state.get('time_convertible_variable')) and reward > 0


def test_setting_none_to_state_attr(dummy_time_series_data) -> None:
    """
    Tests if action sets None in time_convertible_variable attr
    if there's no time convertable variable
    """
    state = State()
    dummy_time_series_data['dummy_col'] = dummy_time_series_data['result'] + 1
    state, reward, _ = time_convertible_variable_search(
        state, dummy_time_series_data[['dummy_col', 'result']])
    assert state.get('time_convertible_variable') is None and reward > 0


def test_actions_yields_negative_reward_if_state_is_unchanged(dummy_time_series_data) -> None:
    """
    Tests if action sets None in time_convertible_variable attr
    if there's no time convertable variable
    """
    state = State()
    time_convertible_variable_search(state, dummy_time_series_data)
    second_run_reward = time_convertible_variable_search(
        state, dummy_time_series_data)[1]
    assert second_run_reward < 0
