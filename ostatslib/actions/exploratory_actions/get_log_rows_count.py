"""
get_log_rows_count module
"""

import numpy as np
from pandas import DataFrame
from ostatslib.actions.utils import ActionResult
from ostatslib.states import State

ROW_COUNT_UPPER_LIMIT = 150000
LOG_COMPRESSION_CONSTANT = 12


def get_log_rows_count(state: State,
                       data: DataFrame) -> ActionResult[None]:
    """
    Gets log rows count: log(#rows)/c, where c is a compression constant

    Args:
        state (State): state
        data (DataFrame): data

    Returns:
        ActionResult[None]: action result
    """
    log_rows_count = __calculate_log_rows_count(data)
    reward = __calculate_reward(state, log_rows_count)
    state = __update_state(state, log_rows_count)
    return ActionResult(state, reward, "get_log_rows_count")


def __calculate_log_rows_count(data: DataFrame) -> float:
    rows_count = len(data.index)

    if rows_count > ROW_COUNT_UPPER_LIMIT:
        return 1

    return np.log10(rows_count)/LOG_COMPRESSION_CONSTANT


def __calculate_reward(state: State, log_rows_count: float) -> float:
    if state.get("log_rows_count") == log_rows_count:
        return -1

    return 0.75


def __update_state(state: State, log_rows_count: float) -> State:
    state.set("log_rows_count", log_rows_count)
    return state
