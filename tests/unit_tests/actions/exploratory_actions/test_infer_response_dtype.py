# pylint: disable=redefined-outer-name
"""
infer_response_dtype action testing module
"""

from pandas import DataFrame
from pandas.api.types import infer_dtype
import pytest

from ostatslib.actions.exploratory_actions import infer_response_dtype
from ostatslib.states import State


def test_yields_negative_reward_if_there_is_no_response_var() -> None:
    """
    In order to infer response dtype, there must be a response variable
    """
    data = DataFrame({'test': [1, 2, 3]})
    state = State()
    state.set('response_variable_label', "")
    reward = infer_response_dtype(state, data)[1]
    assert reward < 0


def test_update_state_with_inferred_dtype() -> None:
    """
    state response_inferred_dtype must be updated with inferred info
    """
    data = DataFrame({'result': [1, 2, 3]})
    inferred_dtype = infer_dtype(data.result)
    state = State()
    state, reward, _ = infer_response_dtype(state, data)
    assert reward > 0
    assert state.get('response_inferred_dtype') == inferred_dtype

def test_yields_negative_reward_and_flags_exception_if_invalid_response_var() -> None:
    """
    Invalid response variable label should yield negative reward 
    and have exception flagged in info
    """
    data = DataFrame({'test': [1, 2, 3]})
    state = State()
    _, reward, info = infer_response_dtype(state, data)
    assert reward < 0
    assert info['raised_exception']
