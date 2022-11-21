"""
Environment module
"""

from copy import deepcopy
from numpy import ndarray
from pandas import DataFrame
from ostatslib.actions import ActionsSpace
from ostatslib.actions.utils import ActionResult
from ostatslib.states import State


class Environment:
    """
    Statistical environment
    """

    def __init__(self, actions_space: ActionsSpace = None) -> None:
        self.__actions_space = actions_space if actions_space is not None else ActionsSpace()

    @property
    def actions_space(self) -> ActionsSpace:
        """
        Gets ActionsSpace

        Returns:
            ActionsSpace: actions space
        """
        return self.__actions_space

    def run_action(self,
                   state: State,
                   data: DataFrame,
                   action_code: ndarray | int) -> tuple[ActionResult, bool]:
        """
        Runs action

        Args:
            state (State): state
            data (DataFrame): data
            action_code (ndarray): action to be executed

        Returns:
            tuple[ActionResult, bool]: ActionResult and done flag
        """
        action_fn = None
        if isinstance(action_code, ndarray):
            action_fn = self.__actions_space.get_action_by_encoding(
                action_code)
        else:
            action_name = self.__actions_space.actions_names_list[action_code]
            action_fn = self.actions_space.get_action_by_name(action_name)

        action_result = action_fn(deepcopy(state), data)
        done = self.__is_done(action_result.state)
        return action_result, done

    def __is_done(self, state: State) -> bool:
        return bool(state.get("score"))
