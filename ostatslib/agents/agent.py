"""
Open Stats Agent abstract module
"""

from abc import ABC, abstractmethod
from typing import Tuple
from numpy import ndarray
from pandas import DataFrame

from ostatslib.agents.analysis_result import AnalysisResult
from ostatslib.environments import GymEnvironment
from ostatslib.states.state import State


class Agent(ABC):
    """
    Open Stats Agent abstract class
    """

    @abstractmethod
    def train(self, steps: int = 100000) -> None:
        """
        Trains an agent

        Args:
            steps (int, optional): Maximum number of steps during training. Defaults to 100e3.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Saves agent prediction model

        Args:
            path (str): path to file
        """

    @abstractmethod
    def _predict(self, observation: dict) -> ndarray:
        """
        Gets predicted action from agent's model

        Args:
            observation (dict): environment observation

        Returns:
            ndarray: action
        """

    def analyze(self,
                data: DataFrame,
                initial_state: State = State(),
                max_steps: int = 20) -> AnalysisResult:
        """
        Analyzes a dataset

        Args:
            data (DataFrame): dataset
            initial_state (State, optional): initial state. Defaults to State().
            max_steps (int, optional): maximum number of steps. Defaults to 20.

        Returns:
            AnalysisResult: analysis result
        """
        environment = GymEnvironment()
        environment.reset()
        environment.set_data(data)
        environment.set_state(initial_state)
        actions_results = []
        observation = initial_state.features_dict
        terminated = False

        for _ in range(max_steps):
            action = self._predict(observation)
            observation, terminated, truncated, info = self._step(environment,
                                                                  action)
            actions_results.append(info['action_result'])

            if terminated or truncated:
                break

        environment.reset()
        return AnalysisResult(initial_state, actions_results, terminated)

    def _step(self, environment: GymEnvironment, action: ndarray) -> Tuple[dict, bool, bool, dict]:
        observation, _, terminated, truncated, info = environment.step(action)
        return observation, terminated, truncated, info
