"""
time_convertable_variable_search module
"""

from pandas import DataFrame
from pandas.api.types import infer_dtype
from ostatslib.actions.utils import ActionResult, split_response_from_explanatory_variables
from ostatslib.states import State

ORDERED_LIST_OF_POSSIBLE_TIME_DTYPES = [
    "datetime64", "datetime", "date",
    "period",
    "time",
    "timedelta64", "timedelta",
]


def time_convertable_variable_search(state: State,
                                     data: DataFrame) -> ActionResult[str]:
    """
    Gets log rows count: log(#rows)/c, where c is a compression constant

    Args:
        state (State): state
        data (DataFrame): data

    Returns:
        ActionResult[None]: action result
    """
    variables_dataframe: DataFrame = split_response_from_explanatory_variables(state,
                                                                               data)[1]
    date_convertable_variable = __date_variable_search(variables_dataframe)

    reward: float = __calculate_reward(state, date_convertable_variable)
    state: State = __update_state(state, date_convertable_variable)
    return ActionResult(state, reward, "time_convertable_variable")


def __date_variable_search(variables_dataframe: DataFrame) -> str | None:
    time_related_variables: list[tuple[str, str]] = []

    for var_name, values in variables_dataframe.items():
        inferred_dtype_name: str = infer_dtype(values)
        if inferred_dtype_name in ORDERED_LIST_OF_POSSIBLE_TIME_DTYPES:
            time_related_variables.append((var_name, inferred_dtype_name))

    if not time_related_variables:
        return None

    if len(time_related_variables) == 1:
        return time_related_variables[0][0]

    return __select_best_time_related_variable(time_related_variables)


def __select_best_time_related_variable(time_related_variables) -> str:
    for time_dtype in ORDERED_LIST_OF_POSSIBLE_TIME_DTYPES:
        for var_name, inferred_dtype_name in time_related_variables:
            if time_dtype == inferred_dtype_name:
                return var_name

    return time_related_variables[0]


def __calculate_reward(state: State, date_convertable_variable: str | None) -> float:
    if state.get("time_convertable_variable") == date_convertable_variable:
        return -1

    if date_convertable_variable == "":
        return 0.5

    return 0.75


def __update_state(state: State, date_convertable_variable: str | None) -> State:
    state.set("time_convertable_variable", date_convertable_variable)
    return state
