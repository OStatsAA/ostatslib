"""
Upper Confidence Bounds module
https://lilianweng.github.io/posts/2020-06-07-exploration-drl/
"""

from ostatslib.reinforcement_learning_models import Model
from ostatslib.replay_memories import ReplayMemory
from ostatslib.states.state import State
from .exploration_strategy import ExplorationStrategy


class UpperConfidenceBounds(ExplorationStrategy):
    """
    TODO
    """

    def get_action(self,
                   model: Model,
                   state: State,
                   actions_list: list[str],
                   agent_memory: ReplayMemory) -> str:
        return ""
