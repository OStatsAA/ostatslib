"""
is_response_discrete_check module
"""

from copy import deepcopy
from pandas import DataFrame, Series
import numpy as np

from ostatslib.actions.utils import ActionResult
from ostatslib.states import State


def is_response_discrete_check(state: State, data: DataFrame) -> ActionResult[str]:
    """
    Check if response variable is discrete

    Args:
        state (State): state
        data (DataFrame): data

    Returns:
        ActionResult[str]: action result
    """
    state_copy: State = deepcopy(state)
    response_var_label: str = state.get("response_variable_label")
    response: Series = data[response_var_label]

    is_discrete: bool = __is_discrete_check(response)
    if is_discrete:
        state.set("is_response_discrete", 1)
    else:
        state.set("is_response_discrete", -1)

    reward = __calculate_reward(state, state_copy)
    return ActionResult(state, reward, "is_response_discrete_check")


def __is_discrete_check(values: Series) -> bool:
    unique_values = values.unique()

    is_numeric = np.issubdtype(unique_values.dtype, np.number)
    if not is_numeric:
        return False

    is_inexact = np.issubdtype(unique_values.dtype, np.inexact)
    if is_inexact:
        for value in unique_values:
            is_whole = float(value).is_integer()
            if not is_whole:
                return False

    return True


def __calculate_reward(state: State, state_copy: State) -> float:
    if state == state_copy:
        return -1

    return 0.75
