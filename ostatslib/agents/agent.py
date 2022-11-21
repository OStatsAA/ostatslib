"""
Agent module
"""

from pandas import DataFrame

from ostatslib.environments import Environment
from ostatslib.reinforcement_learning_methods import ReinforcementLearningMethod, ActorCritic
from ostatslib.states import State


class Agent:
    """
    Agent class
    """

    def __init__(self,
                 environment: Environment = None,
                 rl_method: ReinforcementLearningMethod = None,
                 is_training: bool = False,
                 max_steps: int = 10) -> None:
        self.__env = environment if environment is not None else Environment()
        self.__rl_method = rl_method if rl_method is not None else ActorCritic()
        self.__is_training = is_training
        self.__max_steps = max_steps

    @property
    def is_training(self) -> bool:
        """
        Return whether agent is in training mode

        Returns:
            bool: is_training flag
        """
        return self.__is_training

    @property
    def rl_method(self) -> ReinforcementLearningMethod:
        """
        Gets agent's Reinforcement Learning Method

        Returns:
            ReinforcementLearningMethod: Reinforcement learning method
        """
        return self.__rl_method

    @property
    def environment(self) -> Environment:
        """
        Gets agent's Statistical Environment

        Returns:
            Environment: statistical environment
        """
        return self.__env

    def train(self, data: DataFrame, initial_state: State = State()) -> float:
        """
        Run one training episode

        Args:
            data (DataFrame): data
            initial_state (State, optional): initial state. Defaults to State().

        Returns:
            float: episode reward
        """
        self.__is_training = True
        episode_reward = self.__rl_method.run_training_episode(initial_state,
                                                               data,
                                                               self.__env,
                                                               self.__max_steps)
        self.__is_training = False
        return episode_reward

    def run_analysis(self, data: DataFrame, initial_state: State = State()) -> list:
        """
        Run an analysis

        Args:
            data (DataFrame): data
            initial_state (State, optional): initial state. Defaults to State().

        Returns:
            list: list of actions take to analyise input data
        """
        episode_actions = self.__rl_method.run_analysis(initial_state,
                                                        data,
                                                        self.__env,
                                                        self.__max_steps)
        return episode_actions
