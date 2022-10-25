"""
ActionResult module
"""

from typing import Generic, TypeVar
from ostatslib.states import State

T = TypeVar("T")


class ActionResult(Generic[T]):
    """
    Class holding state, reward and return object after an executed action

    Args:
        Generic (_type_): generic resulting object
    """

    def __init__(self, state: State, reward: float, result: T) -> None:
        self.state = state
        self.reward = reward
        self.result = result
