"""
ActionsSpace module
"""

from typing import TypeVar
from ostatslib.actions.exploratory_actions import (
    get_log_rows_count,
    get_response_variable_type
)
from ostatslib.actions.regression_models import linear_regression
from ostatslib.actions.classifiers import logistic_regression
from ostatslib.actions.utils import ActionFunction

T = TypeVar("T")

CLASSIFIERS = {
    'logistic_regression': logistic_regression
}

EXPLORATORY_ACTIONS = {
    'get_log_rows_count': get_log_rows_count,
    'get_response_variable_type': get_response_variable_type
}

REGRESSION_MODELS = {
    'linear_regression': linear_regression
}


class ActionsSpace:
    """
    Actions space
    """

    def __init__(self) -> None:
        self.__actions = EXPLORATORY_ACTIONS | CLASSIFIERS | REGRESSION_MODELS

    @property
    def actions(self) -> dict[str, ActionFunction]:
        """
        Gets actions dictionary

        Returns:
            dict: actions dictionary
        """
        return self.__actions

    @property
    def actions_names_list(self) -> list[str]:
        """Gets actions names list (keys in actions dictionary)

        Returns:
            list[str]: actions names
        """
        return list(self.__actions.keys())

    def get_action(self, action_name: str) -> ActionFunction[T]:
        """
        Gets action function

        Args:
            action_name (str): action name

        Returns:
            ActionFunction[T]: action function
        """
        return self.__actions[action_name]

    def __len__(self):
        return len(self.__actions)
