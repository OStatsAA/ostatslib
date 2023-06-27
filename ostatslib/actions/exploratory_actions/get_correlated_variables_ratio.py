"""
get_correlated_variables module
"""

import numpy as np
from pandas import DataFrame
from ostatslib.states import State

from ._get_exploratory_reward import get_exploratory_reward
from ..action import Action, ActionInfo, ActionResult

_ACTION_NAME = "Get Correlated Variables Ratio"

CORRELATION_THRESHOLD = 0.5


def _action(state: State, data: DataFrame) -> ActionResult[None]:
    """
    Gets correlated variables ratio using correlation matrix

    Args:
        state (State): state
        data (DataFrame): data

    Returns:
        ActionResult[None]: action result
    """
    state_copy = state.copy()
    corr_matrix = data.corr()
    correlated_variables_ratio = __get_correlated_ratio(corr_matrix)
    __update_state(state, correlated_variables_ratio)
    reward = get_exploratory_reward(state, state_copy)
    return state, reward, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_action,
                                     model=None,
                                     raised_exception=False)


def __get_correlated_ratio(corr_matrix):
    above_threshold_matrix = corr_matrix.abs() > CORRELATION_THRESHOLD
    return np.tril(above_threshold_matrix, -1).sum()/corr_matrix.shape[0]


def __update_state(state: State, correlated_variables_ratio: float) -> State:
    if not correlated_variables_ratio:
        state.set("correlated_variables_ratio", -1)
        return state

    state.set("correlated_variables_ratio", correlated_variables_ratio)
    return state


get_correlated_variables_ratio: Action[None] = _action
