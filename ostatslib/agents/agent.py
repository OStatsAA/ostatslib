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
                 model: Model = None,
                 is_training: bool = False,
                 exploration_strategy: ExplorationStrategy = None,
                 replay_memory: ReplayMemory = None,
                 learning_rate: float = .01,
                 discount: float = 1) -> None:

        self.__model = model if model is not None else SupportVectorRegression()
        self.is_training = is_training
        self.__exploration_strategy = (
            exploration_strategy if exploration_strategy is not None else EpsilonGreedy())
        self.__memory = replay_memory if replay_memory is not None else ReplayMemory()
        self.__learning_rate = learning_rate
        self.__discount = discount

    def remember_transition(self,
                            state: State,
                            action_code: np.ndarray,
                            next_state: State,
                            reward: float,
                            available_actions: np.ndarray) -> None:
        """
        Stores transition in agent's memory

        Args:
            state (State): state
            action_name (str): action taken name
            next_state (State): resulting state
            reward (float): reward received
        """
        state_qvalue = (1 - self.__learning_rate) * self.get_qvalue(state, action_code)
        next_state_qvalues = self.get_qvalue(next_state, available_actions)
        next_state_best_qvalue = np.amax(next_state_qvalues)
        discounted_next_state_qvalue = self.__discount * next_state_best_qvalue
        target_qvalue = reward + discounted_next_state_qvalue

        state_qvalue += self.__learning_rate * (target_qvalue - state_qvalue)
        self.__memory.append(state.features_vector,
                             action_code,
                             next_state.features_vector,
                             state_qvalue)

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
        self.__model.fit(*self.__memory.get_sar_entries().values())

    def get_qvalue(self, state: State, action: np.ndarray) -> np.ndarray:
        """
        Returns QValues approximated by agent's model

        Args:
            state (State): state
            action (np.ndarray): action(s)

        Returns:
            np.ndarray: QValues for action(s)
        """
        if self.__model.is_fit:
            return self.__model.predict(state.features_vector, action)

        return 0

    def get_action(self, state: State, available_actions: np.ndarray) -> np.ndarray:
        """
        Gets an action

        Args:
            state (State): state

        Returns:
            str: action name
        """
        if not self.is_training:
            predictions = self.__model.predict(state.features_vector,
                                               available_actions)
            return available_actions[np.argmax(predictions)]

        return self.__exploration_strategy.get_action(self.__model,
                                                      state,
                                                      available_actions,
                                                      self.__memory)
