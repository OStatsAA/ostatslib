import inspect
import numpy as np
from pandas import DataFrame

import pytest
from ostatslib.actions.base import Action, ExploratoryAction
import ostatslib.actions.exploratory_actions.metrics_exploratory_actions as metrics_expl_actions
from ostatslib.config import DEFAULT_CONFIG
from ostatslib.states import State


def _is_action(type_: type):
    return inspect.isclass(type_) and issubclass(type_, Action) and not type_ == ExploratoryAction


@pytest.mark.parametrize('action',
                         [action[1]() for action in inspect.getmembers(
                             metrics_expl_actions, _is_action)],
                         ids=[str(action[0]) for action in inspect.getmembers(metrics_expl_actions, _is_action)])
def test_metric_exploratory_actions_update_keys_in_state(action: Action) -> None:
    """
    Tests if exploratory actions update state using action_key attribute
    """
    init_state = State()
    init_state.set('response_variable_label', 'target')
    data = DataFrame({'x': np.ones(10), 'target': np.random.rand(10)})
    key = action.action_key

    next_state, *_ = action.execute(data, init_state.copy(), DEFAULT_CONFIG)

    assert init_state.get(key) != next_state.get(key)


def test_missing_data_ratio_offset() -> None:
    """Tests if missing data ratio is higher than offset if any NaNs are found
    """
    action = metrics_expl_actions.MissingDataRatioExploration()
    init_state = State()
    init_state.set('response_variable_label', 'target')
    data = DataFrame({'x': np.ones(100), 'target': np.random.rand(100)})
    data.iloc[1]['x'] = None
    key = action.action_key

    next_state, *_ = action.execute(data, init_state.copy(), DEFAULT_CONFIG)

    assert next_state.get(key) > getattr(action, '_OFFSET')
