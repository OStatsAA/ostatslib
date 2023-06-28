"""
get_log_columns_count module
"""

import numpy as np
from pandas import DataFrame
from ostatslib import config
from ostatslib.states import State

from ..action import Action, ActionInfo, ActionResult

_ACTION_NAME = "Get Log Columns Count"


COLUMNS_COUNT_UPPER_LIMIT = 1000


def _action(state: State, data: DataFrame) -> ActionResult[None]:
    """
    Gets log columns count: log_1000(#columns),

    Args:
        state (State): state
        data (DataFrame): data

    Returns:
        ActionResult[None]: action result
    """
    log_columns_count = __calculate_log_columns_count(data)
    reward = __calculate_reward(state, log_columns_count)
    state = __update_state(state, log_columns_count)
    return state, reward, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_action,
                                     model=None,
                                     raised_exception=False)


def __calculate_log_columns_count(data: DataFrame) -> float:
    columns_count = len(data.columns)

    if columns_count > COLUMNS_COUNT_UPPER_LIMIT:
        return 1

    return np.emath.logn(COLUMNS_COUNT_UPPER_LIMIT, columns_count)


def __calculate_reward(state: State, log_columns_count: float) -> float:
    if state.get("log_columns_count") == log_columns_count:
        return config.MIN_REWARD

    return config.MAX_EXPLORATORY_REWARD


def __update_state(state: State, log_columns_count: float) -> State:
    state.set("log_columns_count", log_columns_count)
    return state


get_log_columns_count: Action[None] = _action
