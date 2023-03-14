"""
GymEnvironment module
"""

from typing import Any
from numpy import ndarray
from pandas import DataFrame
from gymnasium import Env
from gymnasium.spaces import Dict
from ostatslib.actions import ActionsSpace
from ostatslib.actions.utils.action_result import ActionResult
from ostatslib.states import State
from .utils import generate_training_dataset

ObsType = Dict
REWARD_RANGE = (-1, 1)


class GymEnvironment(Env):
    """
    Statistical environment implemented as Gymnasium environment
    """

    __state: State
    __data: DataFrame

    def __init__(self) -> None:
        self.__init_state_and_data()
        self.observation_space = State().as_gymnasium_space()
        self.action_space: ActionsSpace = ActionsSpace()
        self.reward_range = REWARD_RANGE
        self.__steps_taken = 0

    def render(self) -> None:
        print("Render has no effect yet")

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[dict, dict]:
        self.__init_state_and_data()
        self.__steps_taken = 0
        super().reset(seed=seed, options=options)
        return self.__state.features_dict, dict({'action_result': None})

    def step(self, action: ndarray) -> tuple[dict, float, bool, bool, dict]:
        action_fn = self.action_space.get_action_by_encoding(action)
        action_result = ActionResult(self.__state, -1, "Invalid")
        if action_fn is not None:
            action_result = action_fn(self.__state.copy(), self.__data)
            self.__steps_taken += 1

        self.__state = action_result.state
        observation = action_result.state.features_dict
        terminated = self.__is_done(action_result.state)

        return (observation,
                action_result.reward,
                terminated,
                False,
                dict({'action_result': action_result}))

    def set_data(self, data: DataFrame) -> None:
        """
        Set dataset to be used until next reset

        Args:
            data (DataFrame): dataset
        """
        self.__data = data

    def set_state(self, state: State) -> None:
        """
        Set analysis state

        Args:
            state (State): State
        """
        self.__state = state

    def __is_done(self, state: State) -> bool:
        return bool(state.get("score") > 0.6) or self.__steps_taken >= 10

    def __init_state_and_data(self):
        self.__data, self.__state = generate_training_dataset()
