"""
Agent module
"""

from ostatslib.agents.model import Model
from ostatslib.replay_memories.replay_memory import ReplayMemory
from ostatslib.states.state import State
from ostatslib.exploration_strategies.epsilon_greedy import EpsilonGreedy
from ostatslib.exploration_strategies.exploration_strategy import ExplorationStrategy


class Agent:
    """
    Agent class
    """

    def __init__(self,
                 actions: list[str],
                 model: Model = None,
                 is_training: bool = False,
                 exploration_strategy: ExplorationStrategy = EpsilonGreedy(),
                 replay_memory: ReplayMemory = ReplayMemory()) -> None:

        self.__actions = actions
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

    def is_memory_full(self) -> bool:
        """
        Return whether memory is full

        Returns:
            bool: whether memory is full or not
        """
        return self.__memory.is_full()

    def update_model(self) -> None:
        """
        Updates model used to estimate best actions
        """
        self.__model.fit(self.__memory)

    def get_action(self, state: State) -> str:
        """
        Gets an action

        Args:
            state (State): state

        Returns:
            str: action name
        """
        if not self.__is_training:
            return self.__model.predict(state)

        return self.__exploration_strategy.get_action(self.__model,
                                                      state,
                                                      self.__actions,
                                                      self.__memory)
