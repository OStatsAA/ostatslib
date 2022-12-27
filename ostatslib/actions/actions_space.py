"""
ActionsSpace module
"""

from functools import cached_property
from typing import TypeVar
import numpy as np
from ostatslib.actions.exploratory_actions import (
    get_log_rows_count,
    is_response_dichotomous_check,
    is_response_discrete_check,
    is_response_positive_values_only_check,
    is_response_quantitative_check,
)
from ostatslib.actions.regression_models import (
    linear_regression,
    poisson_regression,
    support_vector_regression
)
from ostatslib.actions.classifiers import (
    logistic_regression,
    support_vector_classification,
    decision_tree
)
from ostatslib.actions.utils import ActionFunction, as_binary_array

T = TypeVar("T")

ENCODING_LENGTH = 5

# Encoding: 0 to 7
EXPLORATORY_ACTIONS = {
    'get_log_rows_count': (get_log_rows_count,
                           as_binary_array(0, ENCODING_LENGTH)),
    'is_response_dichotomous_check': (is_response_dichotomous_check,
                                      as_binary_array(1, ENCODING_LENGTH)),
    'is_response_discrete_check': (is_response_discrete_check,
                                   as_binary_array(2, ENCODING_LENGTH)),
    'is_response_positive_values_only_check': (is_response_positive_values_only_check,
                                               as_binary_array(3, ENCODING_LENGTH)),
    'is_response_quantitative_check': (is_response_quantitative_check,
                                       as_binary_array(4, ENCODING_LENGTH))
}

# Encoding: 8 to 15
CLASSIFIERS = {
    'logistic_regression': (logistic_regression,
                            as_binary_array(8, ENCODING_LENGTH)),
    'support_vector_classification': (support_vector_classification,
                                      as_binary_array(9, ENCODING_LENGTH)),
    'decision_tree': (decision_tree,
                      as_binary_array(10, ENCODING_LENGTH))
}

# Encoding: 16 to 23
REGRESSION_MODELS = {
    'linear_regression': (linear_regression,
                          as_binary_array(16, ENCODING_LENGTH)),
    'poisson_regression': (poisson_regression,
                           as_binary_array(17, ENCODING_LENGTH)),
    'support_vector_regression': (support_vector_regression,
                                  as_binary_array(18, ENCODING_LENGTH))
}


class ActionsSpace:
    """
    Actions space
    """

    def __init__(self) -> None:
        self.__actions = EXPLORATORY_ACTIONS | CLASSIFIERS | REGRESSION_MODELS

    @cached_property
    def actions(self) -> dict[str, tuple[ActionFunction, np.array]]:
        """
        Gets actions dictionary

        Returns:
            dict: actions dictionary
        """
        return self.__actions

    @cached_property
    def actions_names_list(self) -> list[str]:
        """
        Gets actions names list (keys in actions dictionary)

        Returns:
            list[str]: actions names
        """
        return list(self.__actions.keys())

    @cached_property
    def actions_encodings_list(self) -> np.ndarray:
        """
        Gets actions encodings list

        Returns:
            list[str]: actions names
        """
        actions_array = np.ndarray(shape=(len(self), ENCODING_LENGTH))
        index = 0
        for action_value in self.__actions.values():
            actions_array[index] = action_value[1]
            index += 1

        return actions_array

    @cached_property
    def encoding_length(self) -> int:
        """
        Returns encoding length (# of digits in the)

        Returns:
            int: # of digits in the encoding
        """
        return ENCODING_LENGTH

    def get_action_by_name(self, action_name: str) -> ActionFunction[T]:
        """
        Gets action function

        Args:
            action_name (str): action name

        Returns:
            ActionFunction[T]: action function
        """
        return self.__actions[action_name][0]

    def get_action_by_encoding(self, action_code: np.array) -> ActionFunction[T]:
        """
        Gets action function

        Args:
            action_code (ndarray): action code

        Returns:
            ActionFunction[T]: action function
        """
        for action in self.__actions.values():
            if np.array_equal(action[1], action_code):
                return action[0]

        raise ValueError(f"Action code not found: {action_code}")

    def __len__(self):
        return len(self.__actions)
