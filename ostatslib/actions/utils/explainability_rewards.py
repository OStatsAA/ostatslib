"""
explainability rewards decorators module
"""

from functools import wraps
from typing import Callable, TypeVar

from pandas import DataFrame
from ostatslib.states.state import State
from .action_result import ActionResult

T = TypeVar("T")

OPAQUE_PENALTY = -10
INTERPETRABLE_REWARD = 10
COMPREHENSIBLE_REWARD = 10


def opaque_model(
        action_function: Callable[[State, DataFrame], ActionResult[T]]) -> ActionResult[T]:
    """
    Penalty for actions resulting in an opaque model

    Args:
        action_function (Callable[[State, DataFrame], ActionResult[T]]): action function

    Returns:
        ActionResult[T]: action results
    """
    wraps(action_function)

    def function_wrapper(*args, **kargs):
        action_result = action_function(*args, **kargs)
        action_result.reward += OPAQUE_PENALTY
        return action_result

    return function_wrapper


def interpretable_model(
        action_function: Callable[[State, DataFrame], ActionResult[T]]) -> ActionResult[T]:
    """
    Reward for actions resulting in an interpetrable model

    Args:
        action_function (Callable[[State, DataFrame], ActionResult[T]]): action function

    Returns:
        ActionResult[T]: action results
    """
    wraps(action_function)

    def function_wrapper(*args, **kargs):
        action_result = action_function(*args, **kargs)
        action_result.reward += INTERPETRABLE_REWARD
        return action_result

    return function_wrapper


def comprehensible_model(
        action_function: Callable[[State, DataFrame], ActionResult[T]]) -> ActionResult[T]:
    """
    Reward for actions resulting in an comprehensible model

    Args:
        action_function (Callable[[State, DataFrame], ActionResult[T]]): action function

    Returns:
        ActionResult[T]: action results
    """
    wraps(action_function)

    def function_wrapper(*args, **kargs):
        action_result = action_function(*args, **kargs)
        action_result.reward += COMPREHENSIBLE_REWARD
        return action_result

    return function_wrapper