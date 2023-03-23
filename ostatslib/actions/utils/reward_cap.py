"""
reward_cap function module
"""

from functools import wraps
from typing import TypeVar
from pandas import DataFrame
from ostatslib.actions import Action, TModel
from ostatslib.states import State

REWARD_UPPER_LIMIT = float(1)
REWARD_LOWER_LIMIT = float(-1)

T = TypeVar('T')

def reward_cap(action_function: Action[TModel]) -> Action[TModel]:
    """
    Limits rewards from an action within lower and upper limits

    Args:
        action_function (Action[TModel]): action

    Returns:
        Action[TModel]: action
    """
    wraps(action_function)

    def function_wrapper(state: State, data: DataFrame):
        state, reward, info = action_function(state, data)
        reward = min(max(REWARD_LOWER_LIMIT, reward), REWARD_UPPER_LIMIT)
        return state, reward, info

    return function_wrapper
