import numpy as np
from pandas import DataFrame, date_range

import pytest
from ostatslib.actions.exploratory_actions.time_convertible_variable_search import TimeConvertibleVariableSearch
from ostatslib.config import DEFAULT_CONFIG
from ostatslib.states import State

datasets = [
    DataFrame({'x': np.ones(10),
               'time': date_range('1970-01-01', periods=10, freq='Y'),
              'target': np.random.rand(10)}),
    DataFrame({'x': np.ones(10),
               'time': date_range('1970-01-01', periods=10, freq='Y'),
               'time2': date_range('1970-01-01', periods=10, freq='D'),
              'target': np.random.rand(10)})
]


@pytest.mark.parametrize('data', datasets, ids=['1 Time Var', '2 Time Vars'])
def test_time_convertible_finds_time_variable(data: DataFrame) -> None:
    """
    Tests if time convertible search finds datetime variable in data
    """
    action = TimeConvertibleVariableSearch()
    init_state = State()
    init_state.set('response_variable_label', 'target')

    next_state, *_ = action.execute(data, init_state.copy(), DEFAULT_CONFIG)

    assert next_state.get('time_convertible_variable')


def test_time_convertible_sets_time_var_to_none_in_state() -> None:
    """
    Tests if time convertible search sets time var to None if none if found
    """
    action = TimeConvertibleVariableSearch()
    init_state = State()
    init_state.set('response_variable_label', 'target')
    data = DataFrame({'x': np.ones(10), 'target': np.random.rand(10)})
    next_state, *_ = action.execute(data, init_state.copy(), DEFAULT_CONFIG)

    assert next_state.get('time_convertible_variable') is None
