"""
GymEnvironment module
"""

from typing import Any
from numpy import ndarray
from gymnasium import Env
from gymnasium.spaces import Dict
from ostatslib.actions import ActionsSpace
from ostatslib.actions.utils import ActionResult
from ostatslib.states import State
from .utils import generate_training_dataset

ObsType = Dict
REWARD_RANGE = (-1, 1)


class GymEnvironment(Env):
    """
    Statistical environment implemented as Gymnasium environment
    """

    __state = None
    __data = None

    def __init__(self) -> None:
        self.__init_state_and_data()
        self.observation_space = State().as_gymnasium_space()
        self.action_space: ActionsSpace = ActionsSpace()
        self.reward_range = REWARD_RANGE

    def render(self) -> None:
        print("Render has no effect yet")

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> None:
        self.__init_state_and_data()
        return super().reset(seed=seed, options=options)

    def step(self, action: ndarray) -> tuple[dict, float, bool, bool, ActionResult]:
        action_fn = self.action_space.get_action_by_encoding(action)
        action_result = action_fn(self.__state.copy(), self.__data)

        self.__state = action_result.state
        observation = action_result.state.features_dict
        terminated = self.__is_done(action_result.state)

        return observation, action_result.reward, terminated, False, action_result

    def __is_done(self, state: State) -> bool:
        return bool(state.get("score") > 0.6)

    def __init_state_and_data(self):
        self.__state = State()
        self.__data = generate_training_dataset()
