"""
is_response_dichotomous_check module
"""

from copy import deepcopy
from pandas import DataFrame, Series
from pandas.api.types import infer_dtype
import numpy as np

from ostatslib.actions.utils import ActionResult
from ostatslib.states import State


def is_response_dichotomous_check(state: State, data: DataFrame) -> ActionResult[str]:
    """
    Check if response variable is dichotomous

    Args:
        state (State): state
        data (DataFrame): data

    Returns:
        ActionResult[str]: action result
    """
    state_copy: State = deepcopy(state)
    response_var_label: str = state.get("response_variable_label")
    response: Series = data[response_var_label]

    is_dichotomous: bool = __is_dichotomous_check(response)
    if is_dichotomous:
        state.set("is_response_dichotomous", 1)
    else:
        state.set("is_response_dichotomous", -1)

    reward = __calculate_reward(state, state_copy)
    return ActionResult(state, reward, "is_response_dichotomous_check")


def __is_dichotomous_check(values: Series) -> bool:
    inferred_dtype: str = infer_dtype(values)
    if inferred_dtype == "boolean":
        return True

    unique_values = values.unique()
    if len(unique_values) > 2:
        return False

    match inferred_dtype:
        case "categorical" | "string" | "object":
            return True
        case "integer":
            return True if __is_within_possible_boolean_range_of_integers(unique_values) else False
        case "floating" | "decimal" | "mixed-integer-float":
            first, second = unique_values
            return True if first.is_integer() and second.is_integer() and \
                __is_within_possible_boolean_range_of_integers(unique_values) else False
        case _:
            return False


def __is_within_possible_boolean_range_of_integers(unique_values):
    return np.any((unique_values >= -1) | (unique_values <= 2))


def __calculate_reward(state: State, state_copy: State) -> float:
    if state == state_copy:
        return -1

    return 0.75
