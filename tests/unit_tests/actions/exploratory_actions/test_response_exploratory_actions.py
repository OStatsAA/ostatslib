import inspect
import numpy as np
from pandas import DataFrame

import pytest
from ostatslib.actions.base import Action, TargetExploratoryAction
import ostatslib.actions.exploratory_actions.response_exploratory_actions as resp_expl_actions
from ostatslib.config import DEFAULT_CONFIG
from ostatslib.states import State


def _is_action(type_: type):
    return inspect.isclass(type_) and issubclass(type_, Action) and not type_ == TargetExploratoryAction


@pytest.mark.parametrize('action',
                         [action[1]() for action in inspect.getmembers(
                             resp_expl_actions, _is_action)],
                         ids=[str(action[0]) for action in inspect.getmembers(resp_expl_actions, _is_action)])
def test_response_exploratory_actions_update_keys_in_state(action: Action) -> None:
    """
    Tests if exploratory actions update state using action_key attribute
    """
    init_state = State()
    init_state.set('response_variable_label', 'target')
    data = DataFrame({'x': np.ones(10), 'target': np.random.rand(10)})
    key = action.action_key

    next_state, *_ = action.execute(data, init_state.copy(), DEFAULT_CONFIG)

    assert init_state.get(key) != next_state.get(key)


@pytest.mark.parametrize('action',
                         [action[1]() for action in inspect.getmembers(
                             resp_expl_actions, _is_action)],
                         ids=[str(action[0]) for action in inspect.getmembers(resp_expl_actions, _is_action)])
def test_response_exploratory_actions_when_no_response_is_set(action: Action) -> None:
    """
    Tests if exploratory actions update state using action_key attribute
    """
    init_state = State()
    data = DataFrame({'x': np.ones(10), 'x2': np.random.rand(10)})

    next_state, reward, _ = action.execute(
        data, init_state.copy(), DEFAULT_CONFIG)

    assert init_state == next_state
    assert reward == DEFAULT_CONFIG['MIN_REWARD']
