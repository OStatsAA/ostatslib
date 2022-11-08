"""
Replay memory module
"""

import numpy as np


class ReplayMemory:
    """
    Dataset containing an agent's memory of past states, actions,\
        next states after action taken and reward.
    """

    def __init__(self, max_length: int = 10000) -> None:
        self.__states = np.empty(1)
        self.__actions = np.empty(1)
        self.__next_states = np.empty(1)
        self.__rewards = np.empty(1)
        self.__next_index = 0
        self.__max_length = max_length

    def append(self,
               state_features: np.ndarray,
               action_code: np.ndarray,
               next_state_features: np.ndarray,
               reward: float) -> None:
        """
        Appends new information to replay memory

        Args:
            state (State): state
            action_name (str): action name
            next_state (State): resulting state
            reward (float): reward received
        """
        if not self.__next_index:
            self.__initialize_arrays(state_features.shape, action_code.shape)

        self.__states[self.__next_index] = state_features
        self.__actions[self.__next_index] = action_code
        self.__next_states[self.__next_index] = next_state_features
        self.__rewards[self.__next_index] = reward
        self.__next_index += 1

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

    def __initialize_arrays(self,
                            states_shape: tuple,
                            actions_shape: tuple) -> None:
        states_memories_shape = (self.__max_length, states_shape[0])
        self.__states = np.ndarray(shape=states_memories_shape)
        self.__next_states = np.ndarray(shape=states_memories_shape)

        actions_memories_shape = (self.__max_length, actions_shape[0])
        self.__actions = np.ndarray(shape=actions_memories_shape)

        self.__rewards = np.ndarray(shape=(self.__max_length, 1))
