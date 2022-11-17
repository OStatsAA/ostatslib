"""
Environment module
"""

from collections import deque
from copy import deepcopy
from typing import Deque
from numpy import ndarray
from pandas import DataFrame
from ostatslib.actions import ActionsSpace
from ostatslib.actions.utils import ActionResult
from ostatslib.agents import Agent
from ostatslib.states import State


class Environment:
    """
    Statistical environment holding available actions states throughout the agent's analysis
    """

    def __init__(self,
                 agent: Agent = None,
                 actions_space: ActionsSpace = None) -> None:
        self.__agent = agent if agent is not None else Agent()
        self.__actions_space = actions_space if actions_space is not None else ActionsSpace()

    @property
    def agent(self) -> Agent:
        """
        Gets agent

        Returns:
            Agent: agent
        """
        return self.__agent

    @property
    def actions_space(self) -> ActionsSpace:
        """
        Gets ActionsSpace

        Returns:
            ActionsSpace: actions space
        """
        return self.__actions_space

    def run_analysis(self,
                     data: DataFrame,
                     initial_state: State = None,
                     max_steps: int = 10) -> tuple[Deque, bool]:
        """
        Run analysis using trained agent

        Args:
            initial_state (State): initial state
            data (DataFrame): data
            max_steps (int, optional): max steps to achieve result. Defaults to 10.

        Returns:
            tuple[Queue, bool]: actions taken and done flag
        """
        state = initial_state if initial_state is not None else State()
        self.__agent.is_training = False
        actions_deque = deque(maxlen=max_steps)
        available_actions = self.__actions_space.actions_encodings_list
        done = False

        for _ in range(max_steps):

            action_fn, _ = self.__get_action(state, available_actions)
            action_result = action_fn(deepcopy(state), data)

            actions_deque.append(action_result)
            state = action_result.state

            done = self.__is_done(state)
            if done:
                break

        return actions_deque, done

    def train_agent(self,
                    data: DataFrame,
                    initial_state: State = None,
                    max_steps: int = 10,
                    ) -> tuple[ActionResult, float, bool, int]:
        """
        Trains an agent given a dataset and its initial state

        Args:
            initial_state (State): initial state
            data (_type_): dataset assigned to train the agent
            agent (Agent): training agent
            max_steps (int, optional): max number of steps the agent can take \
                while analysing data. Defaults to 100.

        Returns:
            Agent: trained agent
        """
        state = initial_state if initial_state is not None else State()
        self.agent.is_training = True
        accumulated_reward: float = 0
        action_result: ActionResult = None
        available_actions = self.__actions_space.actions_encodings_list
        done = False

        for step_number in range(max_steps):

            action_fn, action_code = self.__get_action(state,
                                                       available_actions)
            action_result = action_fn(deepcopy(state), data)

            self.__agent.remember_transition(state,
                                             action_code,
                                             action_result.state,
                                             action_result.reward,
                                             available_actions)

            accumulated_reward += action_result.reward
            state = action_result.state

            done = self.__is_done(state)
            if done:
                break

        self.agent.is_training = False
        return action_result.result, accumulated_reward, done, step_number

    def __get_action(self, state: State, available_actions: ndarray):
        action_code = self.__agent.get_action(state, available_actions)
        action_fn = self.__actions_space.get_action_by_encoding(action_code)
        return action_fn, action_code

    def __is_done(self, state: State) -> bool:
        return bool(state.get("score"))
