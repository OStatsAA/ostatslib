"""
ExplorationStrategy module
"""

from abc import ABC, abstractmethod
from ostatslib.rf_models import Model
from ostatslib.replay_memories.replay_memory import ReplayMemory
from ostatslib.states.state import State


class ExplorationStrategy(ABC):
    """
    Exploration Strategy abstract class
    """

    @abstractmethod
    def get_action(self,
                   model: Model,
                   state: State,
                   actions_list: list[str],
                   agent_memory: ReplayMemory) -> str:
        """
        Gets actions according to exploration strategy

        Args:
            model (object): _description_
            state (Dict): _description_
            actions_list (list[str]): _description_
            agent_memory (Dict): _description_

        Returns:
            str: _description_
        """
