"""
ExplorationStrategy module
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class ExplorationStrategy(ABC):
    """
    Exploration Strategy abstract class
    """

    @abstractmethod
    def get_action(self,
                   model: Any,
                   state: Dict,
                   actions_list: list[str],
                   agent_memory: Dict) -> str:
        """Gets actions according to exploration strategy

        Args:
            model (object): _description_
            state (Dict): _description_
            actions_list (list[str]): _description_
            agent_memory (Dict): _description_

        Returns:
            str: _description_
        """
