"""
Agent module
"""

import numpy as np

from ostatslib.reinforcement_learning_models import Model, SupportVectorRegression
from ostatslib.replay_memories import ReplayMemory
from ostatslib.states import State
from ostatslib.exploration_strategies import EpsilonGreedy, ExplorationStrategy


class Agent:
    """
    Agent class
    """

    def __init__(self,
                 model: Model = SupportVectorRegression(),
                 is_training: bool = False,
                 exploration_strategy: ExplorationStrategy = EpsilonGreedy(),
                 replay_memory: ReplayMemory = ReplayMemory()) -> None:

        self.__model = model
        self.__is_training = is_training
        self.__exploration_strategy = exploration_strategy
        self.__memory = replay_memory

    def remember_transition(self,
                            state: State,
                            action_name: str,
                            next_state: State,
                            reward: float) -> None:
        """
        Stores transition in agent's memory

        Args:
            state (State): state
            action_name (str): action taken name
            next_state (State): resulting state
            reward (float): reward received
        """
        self.__memory.append(state, action_name, next_state, reward)

    @property
    def is_memory_full(self) -> bool:
        """
        Return whether memory is full

        Returns:
            bool: whether memory is full or not
        """
        return self.__memory.is_full()

    @property
    def memory_length(self) -> int:
        """
        Gets memory length

        Returns:
            int: memory length (rows count)
        """
        return len(self.__memory)

    def update_model(self) -> None:
        """
        Updates model used to estimate best actions
        """
        self.__model.fit(self.__memory)

    def get_action(self, state: State, available_actions: np.ndarray) -> np.ndarray:
        """
        Gets an action

        Args:
            state (State): state

        Returns:
            str: action name
        """
        if not self.__is_training:
            return self.__model.predict(state.features_vector, available_actions)

        return self.__exploration_strategy.get_action(self.__model,
                                                      state,
                                                      available_actions,
                                                      self.__memory)
