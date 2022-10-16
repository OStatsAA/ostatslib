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

    def __init__(self, max_length: int = 10000) -> None:
        self.__states = np.empty(max_length, dtype=object)
        self.__actions = np.empty(max_length, dtype=object)
        self.__next_states = np.empty(max_length, dtype=object)
        self.__rewards = np.empty(max_length, dtype=float)
        self.__next_index = 0
        self.__max_length = max_length

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

    def get_free_space_count(self) -> int:
        """
        Returns number of empty rows

        Returns:
            int: number of empty row untill memory is full
        """
        return self.__max_length - self.__next_index

    def is_full(self) -> bool:
        """
        Returns whether the memory is at full capacity or not

        Returns:
            bool: True if memory if full
        """
        return self.__max_length == self.__next_index

    def __len__(self):
        return self.__next_index
