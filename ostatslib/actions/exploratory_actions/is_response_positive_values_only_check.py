# pylint: disable=broad-except
"""
is_response_positive_values_only_check module
"""

from copy import deepcopy
from pandas import DataFrame, Series
import numpy as np

from ostatslib.actions.utils import ActionResult
from ostatslib.states import State


def is_response_positive_values_only_check(state: State, data: DataFrame) -> ActionResult[str]:
    """
    Check if response variable values are positive only

    Args:
        state (State): state
        data (DataFrame): data

    Returns:
        ActionResult[str]: action result
    """
    state_copy: State = deepcopy(state)
    response_var_label: str = state.get("response_variable_label")
    response: Series = data[response_var_label]

    is_positive_only: bool = __positive_only_check(response)
    if is_positive_only:
        state.set("is_response_positive_values_only", 1)
    else:
        state.set("is_response_positive_values_only", -1)

    reward = __calculate_reward(state, state_copy)
    return ActionResult(state, reward, "is_response_positive_values_only_check")


def __positive_only_check(values: Series) -> bool:
    unique_values = values.unique()
    is_numeric = np.issubdtype(unique_values.dtype, np.number)

    if not is_numeric:
        try:
            unique_values.astype(float, copy=False)
        except Exception:
            return False

    return True if unique_values.min() >= 0 else False


def __calculate_reward(state: State, state_copy: State) -> float:
    if state == state_copy:
        return -1

    return 1
