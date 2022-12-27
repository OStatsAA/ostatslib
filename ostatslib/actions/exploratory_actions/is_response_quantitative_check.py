"""
is_response_quantitative_check module
"""

from copy import deepcopy
from pandas import DataFrame, Series
import numpy as np

from ostatslib.actions.utils import ActionResult
from ostatslib.states import State


def is_response_quantitative_check(state: State, data: DataFrame) -> ActionResult[str]:
    """
    Check if response variable is quantitative

    Args:
        state (State): state
        data (DataFrame): data

    Returns:
        ActionResult[str]: action result
    """
    state_copy: State = deepcopy(state)
    response_var_label: str = state.get("response_variable_label")
    response: Series = data[response_var_label]

    is_quantitative: bool = __is_quantitative_check(response)
    if is_quantitative:
        state.set("is_response_quantitative", 1)
    else:
        state.set("is_response_quantitative", -1)

    reward = __calculate_reward(state, state_copy)
    return ActionResult(state, reward, "is_response_quantitative")


def __is_quantitative_check(values: Series) -> bool:
    unique_values = values.unique()
    return np.issubdtype(unique_values.dtype, np.number)


def __calculate_reward(state: State, state_copy: State) -> float:
    if state == state_copy:
        return -1

    return 0.75
