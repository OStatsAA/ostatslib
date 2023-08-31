"""
GymEnvironment module
"""

from typing import Any, Callable
from random import choice
from numpy import ndarray
from pandas import DataFrame
from gymnasium import Env
from gymnasium.spaces import Dict
from ostatslib.config import DEFAULT_CONFIG, Config
from ostatslib.actions import ActionInfo, ActionsSpace
from ostatslib.environments.data_generators import (datacooker_generator,
                                                    pmlb_generator,
                                                    sklearn_generator)
from ostatslib.states import State

ObsType = Dict
DataGeneratorFn = Callable[[], tuple[DataFrame, State]]

DEFAULT_DATA_GENERATORS = [
    datacooker_generator,
    pmlb_generator,
    sklearn_generator]


class GymEnvironment(Env):
    """
    Statistical environment implemented as Gymnasium environment
    """

    _state: State
    _data: DataFrame
    _data_generators: list[DataGeneratorFn]

    def __init__(self,
                 data_generators: list[DataGeneratorFn] | None = None,
                 config: Config = DEFAULT_CONFIG) -> None:
        self.observation_space = State().as_gymnasium_space
        self.action_space: ActionsSpace = ActionsSpace()
        self.config = config
        if data_generators is None:
            self._data_generators = DEFAULT_DATA_GENERATORS
        else:
            self._data_generators = data_generators
        self.__steps_taken = 0
        self.__init_state_and_data()

    def render(self) -> None:
        print("Render has no effect yet")

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[dict, ActionInfo]:
        self.__init_state_and_data()
        self.__steps_taken = 0
        super().reset(seed=seed, options=options)
        return self._state.features_dict, ActionInfo('Invalid Action')

    def step(self, action: ndarray) -> tuple[dict, float, bool, bool, ActionInfo]:
        action_instance = self.action_space.get_action(action)
        if action_instance is None:
            state = self._state
            reward = self.config['MIN_REWARD']
            info = ActionInfo('Invalid Action', next_state=state.copy())
        else:
            state, reward, info = action_instance.execute(self._data,
                                                          self._state,
                                                          self.config)

        self.__steps_taken += 1
        self._state = state
        observation = state.features_dict
        terminated = self.__is_done(state, reward)
        truncated = self.__is_truncated(terminated)

        return observation, reward, terminated, truncated, info

    def set_data(self, data: DataFrame) -> None:
        """
        Set dataset to be used until next reset

        Args:
            data (DataFrame): dataset
        """
        self._data = data

    def set_state(self, state: State) -> None:
        """
        Set analysis state

        Args:
            state (State): State
        """
        self._state = state

    def __is_done(self, state: State, reward: float) -> bool:
        return (float(state.get("score")) > self.config['MIN_ACCEPTED_SCORE'] and
                reward > self.config['MAX_REWARD']/2)

    def __is_truncated(self, terminated: bool) -> bool:
        return (not terminated) and self.__steps_taken >= self.config["MAX_STEPS"]

    def __init_state_and_data(self):
        generator = choice(self._data_generators)
        self._data, self._state = generator()
