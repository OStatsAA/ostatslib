# pylint: disable=broad-except
"""
is_response_positive_values_only_check module
"""

from pandas import DataFrame, Series
import numpy as np
from ostatslib.actions import Action, ActionResult
from ostatslib.states import State


def _is_response_positive_values_only_check(state: State, data: DataFrame) -> ActionResult[None]:
    """
    Check if response variable values are positive only

    Args:
        state (State): state
        data (DataFrame): data

    Returns:
        ActionResult[None]: action result
    """
    state_copy: State = state.copy()
    response_var_label: str = state.get("response_variable_label")
    response: Series = data[response_var_label]

    is_positive_only: bool = __positive_only_check(response)
    if is_positive_only:
        state.set("is_response_positive_values_only", 1)
    else:
        state.set("is_response_positive_values_only", -1)

    reward = __calculate_reward(state, state_copy)
    return state, reward, {'model': None, 'raised_exception': False}


def __positive_only_check(values: Series) -> bool:
    unique_values = values.unique()
    is_numeric = np.issubdtype(unique_values.dtype, np.number)

    if not is_numeric:
        try:
            unique_values.astype(float, copy=False)
        except Exception:
            return False

    return unique_values.min() >= 0


def __calculate_reward(state: State, state_copy: State) -> float:
    if state == state_copy:
        return -1

    return 0.5


is_response_positive_values_only_check: Action[None] = _is_response_positive_values_only_check
