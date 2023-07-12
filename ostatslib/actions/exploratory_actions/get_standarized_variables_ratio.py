"""
get_standarized_variables_ratio module
"""

import numpy as np
from pandas import DataFrame
from ostatslib.states import State

from ._get_exploratory_reward import get_exploratory_reward
from ..action import Action, ActionInfo, ActionResult

_ACTION_NAME = "Get Standarized Variables Ratio"


def _action(state: State, data: DataFrame) -> ActionResult[None]:
    """
    Gets standarized variables ratio

    Args:
        state (State): state
        data (DataFrame): data

    Returns:
        ActionResult[None]: action result
    """
    state_copy = state.copy()
    standarized_variables_ratio = __get_ratio(data)
    __update_state(state, standarized_variables_ratio)
    reward = get_exploratory_reward(state, state_copy)
    return state, reward, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_action,
                                     model=None,
                                     raised_exception=False)


def __get_ratio(data: DataFrame) -> float:
    data_stats = data.describe().loc[['mean', 'std']].T
    standarized_variables_filter = (
        (np.isclose(data_stats['mean'], 0)) &
        (np.isclose(data_stats['std'].round(2), 1))
    )
    standarized_count = data_stats.loc[standarized_variables_filter]
    ratio = standarized_count.shape[0] / data_stats.shape[0]
    return ratio


def __update_state(state: State, standarized_variables_ratio: float) -> State:
    if not standarized_variables_ratio:
        state.set("standarized_variables_ratio", -1)
        return state

    state.set("standarized_variables_ratio", standarized_variables_ratio)
    return state


get_standarized_variables_ratio: Action[None] = _action
