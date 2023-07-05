"""
is_response_balanced_check module
"""

import operator
from pandas import DataFrame, Series
from ostatslib import config

from ostatslib.states import State
from ._get_exploratory_reward import get_exploratory_reward
from ..action import Action, ActionInfo, ActionResult
from ..utils import validate_state

_ACTION_NAME = "Is Response Balanced Check"
_VALIDATIONS = [('response_variable_label', operator.truth, None),
                ('response_unique_values_ratio', operator.lt, 0.25)]


def _action(state: State, data: DataFrame) -> ActionResult[None]:
    """
    Check if response variable is balanced

    Args:
        state (State): state
        data (DataFrame): data

    Returns:
        ActionResult[None]: action result
    """
    if not validate_state(state, _VALIDATIONS):
        return state, config.MIN_REWARD, ActionInfo(action_name=_ACTION_NAME,
                                                    action_fn=_action,
                                                    model=None,
                                                    raised_exception=False)

    state_copy: State = state.copy()
    response_var_label: str = state.get("response_variable_label")
    response: Series = data[response_var_label]
    is_balanced_value = __get_is_balanced_feature_value(response)
    state.set("is_response_balanced", is_balanced_value)
    reward = get_exploratory_reward(state, state_copy)
    return state, reward, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_action,
                                     model=None,
                                     raised_exception=False)


def __get_is_balanced_feature_value(values: Series) -> float:
    min_max_ratio = __get_min_max_ratio(values)
    if min_max_ratio > 0.8:
        return 1
    if min_max_ratio > 0.5:
        return 0.5
    if min_max_ratio > 0.3:
        return -0.5

    return -1


def __get_min_max_ratio(values: Series) -> float:
    unique_values_count = values.value_counts(normalize=True)
    return unique_values_count.min()/unique_values_count.max()


is_response_balanced_check: Action[None] = _action
