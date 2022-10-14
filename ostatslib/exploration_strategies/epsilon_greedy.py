"""
EpsilonGreedy class module
"""

from random import random, choice
from typing import Any, Dict
from .exploration_strategy import ExplorationStrategy


class EpsilonGreedy(ExplorationStrategy):
    """
    Class implementing the epsilon-greedy exploration strategy.\n
    Epsilon = 1 -> randomly selects actions;\n
    Epsilon = 0 -> always selects estimated best action
    """

    def __init__(self, epsilon: float = .5) -> None:
        self.__epsilon = epsilon

    def get_action(self,
                   model: Any,
                   state: dict,
                   actions_list: list[str],
                   agent_memory: Dict) -> str:
        prob = random()
        if prob < self.__epsilon:
            return choice(actions_list)
        else:
            return model.predict(state)
