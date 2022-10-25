"""
reward_cap function module
"""

from functools import wraps
from typing import Callable, TypeVar

from pandas import DataFrame
from ostatslib.states.state import State
from .action_result import ActionResult

T = TypeVar("T")

REWARD_UPPER_LIMIT = 100
REWARD_LOWER_LIMIT = -100


def reward_cap(action_function: Callable[[State, DataFrame], ActionResult[T]]) -> ActionResult[T]:
    """
    Limits rewards from an action

    Args:
        action_function (Callable[[State, DataFrame], ActionResult[T]]): action function

    Returns:
        ActionResult[T]: action results
    """
    wraps(action_function)

    def function_wrapper(*args, **kargs):
        action_result = action_function(*args, **kargs)

        if action_result.reward < REWARD_LOWER_LIMIT:
            action_result.reward = REWARD_LOWER_LIMIT
        elif action_result.reward > REWARD_UPPER_LIMIT:
            action_result.reward = REWARD_UPPER_LIMIT

        return action_result

    return function_wrapper
