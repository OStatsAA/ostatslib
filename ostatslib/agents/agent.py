"""
Open Stats Agent abstract module
"""

from abc import ABC, abstractmethod
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
                initial_state: State = State()) -> AnalysisResult:
        """
        Analyzes a dataset

        Args:
            data (DataFrame): dataset
            initial_state (State, optional): initial state. Defaults to State()

        Returns:
            AnalysisResult: analysis result
        """
        environment = GymEnvironment()
        environment.set_data(data)
        environment.set_state(initial_state)
        analysis_steps = []
        observation = initial_state.features_dict
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = self._predict(observation)
            observation, reward, terminated, truncated, info = environment.step(
                action)
            analysis_steps.append((reward, info))

        environment.reset()
        return AnalysisResult(initial_state, analysis_steps, terminated)
