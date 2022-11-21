"""
ExplorationStrategy module
"""

from abc import ABC, abstractmethod

from numpy import ndarray
from ostatslib.reinforcement_learning_methods import ReinforcementLearningMethod
from ostatslib.replay_memories.replay_memory import ReplayMemory
from ostatslib.states.state import State


class ExplorationStrategy(ABC):
    """
    Exploration Strategy abstract class
    """

    @abstractmethod
    def get_action(self,
                   model: ReinforcementLearningMethod,
                   state: State,
                   actions: ndarray,
                   agent_memory: ReplayMemory) -> ndarray:
        """
        Gets actions according to exploration strategy

        Args:
            model (object): _description_
            state (Dict): _description_
            actions_list (list[str]): _description_
            agent_memory (Dict): _description_

        Returns:
            ndarray: _description_
        """
