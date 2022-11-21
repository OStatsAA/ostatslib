"""
ExplorationStrategy module
"""

from abc import ABC, abstractmethod

from numpy import ndarray
from ostatslib.states.state import State


class ExplorationStrategy(ABC):
    """
    Exploration Strategy abstract class
    """

    @abstractmethod
    def get_action(self,
                   state: State,
                   actions: ndarray,
                   predict_fn: callable) -> ndarray | int:
        """
        Gets actions according to exploration strategy

        Args:
            state (State): _description_
            actions (ndarray): _description_
            predict_fn (callable): _description_

        Returns:
            ndarray | int: _description_
        """
