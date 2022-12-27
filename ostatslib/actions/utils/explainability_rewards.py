"""
explainability rewards decorators module
"""

from functools import wraps
from typing import TypeVar

from pandas import DataFrame

from ostatslib.states import State

from .action_result import ActionFunction, ActionResult

T = TypeVar("T")

OPAQUE_PENALTY = -.1
INTERPETRABLE_REWARD = .1
COMPREHENSIBLE_REWARD = .1


def opaque_model(
        action_function: ActionFunction[T]) -> ActionResult[T]:
    """
    Penalty for actions resulting in an opaque model

    Args:
        action_function (Callable[[State, DataFrame], ActionResult[T]]): action function

    Returns:
        ActionResult[T]: action results
    """
    wraps(action_function)

    def function_wrapper(state: State, data: DataFrame):
        action_result = action_function(state, data)
        action_result.reward += OPAQUE_PENALTY
        return action_result

    return function_wrapper


def interpretable_model(
        action_function: ActionFunction[T]) -> ActionResult[T]:
    """
    Reward for actions resulting in an interpetrable model

    Args:
        action_function (Callable[[State, DataFrame], ActionResult[T]]): action function

    Returns:
        ActionResult[T]: action results
    """
    wraps(action_function)

    def function_wrapper(state: State, data: DataFrame):
        action_result = action_function(state, data)
        action_result.reward += INTERPETRABLE_REWARD
        return action_result

    return function_wrapper


def comprehensible_model(
        action_function: ActionFunction[T]) -> ActionResult[T]:
    """
    Reward for actions resulting in an comprehensible model

    Args:
        action_function (Callable[[State, DataFrame], ActionResult[T]]): action function

    Returns:
        ActionResult[T]: action results
    """
    wraps(action_function)

    def function_wrapper(state: State, data: DataFrame):
        action_result = action_function(state, data)
        action_result.reward += COMPREHENSIBLE_REWARD
        return action_result

    return function_wrapper
