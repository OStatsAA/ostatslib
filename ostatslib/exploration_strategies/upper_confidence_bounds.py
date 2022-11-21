"""
Upper Confidence Bounds module
https://lilianweng.github.io/posts/2020-06-07-exploration-drl/
"""

from numpy import ndarray

from ostatslib.reinforcement_learning_methods import ReinforcementLearningMethod
from ostatslib.replay_memories import ReplayMemory
from ostatslib.states.state import State
from .exploration_strategy import ExplorationStrategy


class UpperConfidenceBounds(ExplorationStrategy):
    """
    TODO
    """

    def get_action(self,
                   model: ReinforcementLearningMethod,
                   state: State,
                   actions: ndarray,
                   agent_memory: ReplayMemory) -> ndarray:
        raise NotImplementedError()
