"""
Model abstract class module
"""

from abc import ABC, abstractmethod

from pandas import DataFrame

from ostatslib.environments import Environment
from ostatslib.states import State


class ReinforcementLearningMethod(ABC):
    """
    Model abstract class
    """

    @property
    @abstractmethod
    def is_fit(self) -> bool:
        """
        Flags if models has already been fitted at least once

        Returns:
            bool: is fit flag
        """

    @abstractmethod
    def fit(self) -> None:
        """
        Fits model
        """

    @abstractmethod
    def run_analysis(self,
                     state: State,
                     data: DataFrame,
                     environment: Environment,
                     max_steps: int) -> tuple[list, bool]:
        """
        Run one episode with n steps, limited to max_steps parameter

        Args:
            state (State): initial state
            data (DataFrame): Dataframe
            environment (Environment): OStatsLib environment
            max_steps (int): máximum number of steps in episode

        Returns:
            tuple[list, bool]: tuple with steps and done flag
        """

    @abstractmethod
    def run_training_episode(self,
                             state: State,
                             data: DataFrame,
                             environment: Environment,
                             max_steps: int) -> float:
        """
        Run one episode with n steps, limited to max_steps parameter and updates internal model

        Args:
            state (State): initial state
            data (DataFrame): Dataframe
            environment (Environment): OStatsLib environment
            max_steps (int): maximum number of steps in episode

        Returns:
            float: reward in episode
        """
