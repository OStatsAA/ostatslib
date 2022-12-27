"""
reward_cap function module
"""

from functools import wraps
from typing import TypeVar

from pandas import DataFrame

from ostatslib.states import State

from .action_result import ActionFunction, ActionResult

T = TypeVar("T")

REWARD_UPPER_LIMIT = 1
REWARD_LOWER_LIMIT = -1


def reward_cap(action_function: ActionFunction[T]) -> ActionResult[T]:
    """
    Limits rewards from an action

    Args:
        action_function (ActionFunction[T]): action function

    Returns:
        ActionResult[T]: action results
    """
    wraps(action_function)

    def function_wrapper(state: State, data: DataFrame):
        action_result = action_function(state, data)

        if action_result.reward < REWARD_LOWER_LIMIT:
            action_result.reward = REWARD_LOWER_LIMIT
        elif action_result.reward > REWARD_UPPER_LIMIT:
            action_result.reward = REWARD_UPPER_LIMIT

        return action_result

    return function_wrapper
