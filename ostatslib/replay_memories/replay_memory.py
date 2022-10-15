"""
Replay memory module
"""

from collections import Counter
import numpy as np
from ostatslib.states.state import State


class ReplayMemory:
    """
    Dataset containing an agent's memory of past states, actions,\
        next states after action taken and reward.
    """

    def __init__(self, length: int = 10000) -> None:
        self.__states = np.empty(length, dtype=object)
        self.__actions = np.empty(length, dtype=object)
        self.__next_states = np.empty(length, dtype=object)
        self.__rewards = np.empty(length, dtype=float)
        self.__next_index = 0

    def append(self,
               state: State,
               action_name: str,
               next_state: State,
               reward: float) -> None:
        """
        Appends new information to replay memory

        Args:
            state (State): state
            action_name (str): action name
            next_state (State): resulting state
            reward (float): reward received
        """
        self.__states[self.__next_index] = state
        self.__actions[self.__next_index] = action_name
        self.__next_states[self.__next_index] = next_state
        self.__rewards[self.__next_index] = reward
        self.__next_index += 1

    def get_actions_count(self) -> Counter:
        """
        Gets actions count in replay memory.

        Returns:
            Counter: Counter
        """
        return Counter(self.__actions[:self.__next_index])

    def get_sar_entries(self) -> dict[str, np.ndarray]:
        """
        Returns dictionary with State, Action and Rewards entries

        Returns:
            dict[str, np.ndarray]: dictionary
        """
        return {
            "states": self.__states[:self.__next_index],
            "actions": self.__actions[:self.__next_index],
            "rewards": self.__rewards[:self.__next_index]
        }

    def __len__(self):
        return self.__next_index
