"""
get_response_variable_type module
"""

from copy import deepcopy
from pandas import DataFrame, Series
from pandas.api.types import infer_dtype

from ostatslib.actions.utils import ActionResult
from ostatslib.states import State


def get_response_variable_type(state: State,
                               data: DataFrame) -> ActionResult[None]:
    """
    Infer about response variable type. Relies on Pandas's infer_dtype function:
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.types.infer_dtype.html

    Args:
        state (State): state
        data (DataFrame): data

    Returns:
        ActionResult[None]: action result
    """
    state_copy = deepcopy(state)
    response_var_label: str = state.get("response_variable_label")
    response = data[response_var_label]

    state = __update_state_response_typing_features(state, response)
    reward = __calculate_reward(state, state_copy)

    return ActionResult(state, reward, "get_response_variable_type")


def __update_state_response_typing_features(state: State, response: Series) -> State:
    unique_values_count: int = response.nunique()
    inferred_dtype: str = infer_dtype(response)

    match inferred_dtype:

        case "boolean":
            state.set("is_response_quantitative", -1)
            state.set("is_response_dichotomous", 1)

        case "floating" | "mixed-integer-float" | "decimal" | "complex" | "integer":
            state.set("is_response_quantitative", 1)
            __set_is_response_dichotomous(state, unique_values_count)

        case "categorical" | "string":
            state.set("is_response_quantitative", -1)
            __set_is_response_dichotomous(state, unique_values_count)

        case _:
            __set_is_response_dichotomous(state, unique_values_count)

    return state

def __set_is_response_dichotomous(state: State, unique_values_count: int) -> None:
    if unique_values_count == 2:
        state.set("is_response_dichotomous", 1)
    else:
        state.set("is_response_dichotomous", -1)


def __calculate_reward(state: State, state_copy: State) -> float:
    if state == state_copy:
        return -1

    return 1
