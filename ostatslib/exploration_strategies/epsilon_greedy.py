"""
EpsilonGreedy class module
"""

from random import random, choice
from ostatslib.rf_models import Model
from ostatslib.replay_memories import ReplayMemory
from ostatslib.states import State
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
                   model: Model,
                   state: State,
                   actions_list: list[str],
                   agent_memory: ReplayMemory) -> str:
        prob = random()
        if prob < self.__epsilon:
            return choice(actions_list)

        return model.predict(state)
