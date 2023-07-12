"""
ActionsSpace module
"""

from functools import cached_property
from gymnasium.spaces import MultiBinary

import numpy as np
import numpy.typing as npt
from pandas import DataFrame
from ostatslib import config

from ostatslib.states import State

from .exploratory_actions import (
    get_log_rows_count,
    infer_response_dtype,
    is_response_dichotomous_check,
    is_response_discrete_check,
    is_response_positive_values_only_check,
    is_response_quantitative_check,
    time_convertible_variable_search,
    get_response_unique_values_ratio,
    get_correlated_variables_ratio,
    get_log_columns_count,
    is_response_balanced_check,
    get_standarized_variables_ratio
)
from .regression_models import (
    linear_regression,
    poisson_regression,
    support_vector_regression,
    decision_tree_regression,
    time_series_auto_arima,
    linear_support_vector_regression,
    random_forest_regression
)
from .classifiers import (
    logistic_regression,
    support_vector_classification,
    decision_tree,
    linear_support_vector_classification,
    random_forest
)
from .clustering import (
    k_means,
    dbscan
)

from .action import Action, ActionInfo, ActionResult


MaskNDArray = npt.NDArray[np.int8]
ENCODING_LENGTH = 6


def _as_binary_array(number: int) -> np.ndarray:
    """
    Converts an integer to a np.array of binary digits

    Args:
        number (int): integer number to be converted

    Returns:
        array: numpy array of converted digits
    """
    binary_str = bin(number)[2:].zfill(ENCODING_LENGTH)

    return np.array([int(digit) for digit in binary_str])


# Encoding: 0 to 31
EXPLORATORY_ACTIONS = {
    'get_log_rows_count': (get_log_rows_count, _as_binary_array(0)),
    'is_response_dichotomous_check': (is_response_dichotomous_check, _as_binary_array(1)),
    'is_response_discrete_check': (is_response_discrete_check, _as_binary_array(2)),
    'is_response_positive_values_only_check': (is_response_positive_values_only_check,
                                               _as_binary_array(3)),
    'is_response_quantitative_check': (is_response_quantitative_check, _as_binary_array(4)),
    'time_convertible_variable_search': (time_convertible_variable_search, _as_binary_array(5)),
    'infer_response_dtype': (infer_response_dtype, _as_binary_array(6)),
    'get_response_unique_values_ratio': (get_response_unique_values_ratio, _as_binary_array(7)),
    'get_correlated_variables_ratio': (get_correlated_variables_ratio, _as_binary_array(8)),
    'get_log_columns_count': (get_log_columns_count, _as_binary_array(9)),
    'is_response_balanced_check': (is_response_balanced_check, _as_binary_array(10)),
    'get_standarized_variables_ratio': (get_standarized_variables_ratio, _as_binary_array(11)),
}

# Encoding: 32 to 39
CLASSIFIERS = {
    'logistic_regression': (logistic_regression, _as_binary_array(32)),
    'support_vector_classification': (support_vector_classification, _as_binary_array(33)),
    'linear_support_vector_classification': (linear_support_vector_classification,
                                             _as_binary_array(34)),
    'decision_tree': (decision_tree, _as_binary_array(35)),
    'random_forest': (random_forest, _as_binary_array(36))
}

# Encoding: 40 to 47
REGRESSION_MODELS = {
    'linear_regression': (linear_regression, _as_binary_array(40)),
    'poisson_regression': (poisson_regression, _as_binary_array(41)),
    'support_vector_regression': (support_vector_regression, _as_binary_array(42)),
    'linear_support_vector_regression': (linear_support_vector_regression, _as_binary_array(43)),
    'decision_tree_regression': (decision_tree_regression, _as_binary_array(44)),
    'random_forest_regression': (random_forest_regression, _as_binary_array(45)),
    'time_series_auto_arima': (time_series_auto_arima, _as_binary_array(46))
}

# Encoding: 48 to 55
CLUSTERING = {
    'k_means': (k_means, _as_binary_array(48)),
    'dbscan': (dbscan, _as_binary_array(49))
}


def _invalid_action_step(state: State, data: DataFrame) -> ActionResult[None]:
    if state and data is not None:
        reward = config.MIN_REWARD
        info = ActionInfo(action_name='Invalid Action',
                          action_fn=_invalid_action_step,
                          model=None,
                          raised_exception=False)
        return state, reward, info

    raise ValueError("State and Data must be valid.")


class ActionsSpace(MultiBinary):
    """
    Actions space
    """

    def __init__(self) -> None:
        self.__actions = EXPLORATORY_ACTIONS | CLASSIFIERS | REGRESSION_MODELS | CLUSTERING
        super().__init__(ENCODING_LENGTH)

    @cached_property
    def actions(self) -> dict[str, tuple[Action, np.ndarray]]:
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
            ndarray: actions codes
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

    def get_action_by_name(self, action_name: str) -> Action:
        """
        Gets action function

        Args:
            action_name (str): action name

        Returns:
            ActionFunction[T]: action function
        """
        return self.__actions[action_name][0]

    def get_action_name_by_code(self, action_code: np.ndarray | list) -> str | None:
        """Gets action name by code

        Args:
            action_code (np.ndarray): action code

        Returns:
            str: action name
        """
        for action_name, (_, code) in self.__actions.items():
            if np.array_equal(code, action_code):
                return action_name

        return None

    def is_valid_action_by_encoding(self, action_code: np.ndarray) -> bool:
        """Check if action code is valid

        Args:
            action_code (np.array): action code

        Returns:
            bool: True if it is valid action code
        """
        for action in self.__actions.values():
            if np.array_equal(action[1], action_code):
                return True

        return False

    def get_action_by_encoding(self, action_code: np.ndarray) -> Action:
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

        return _invalid_action_step

    def sample(self, mask: MaskNDArray | None = None) -> np.ndarray:
        index = np.random.choice(len(self.actions_encodings_list))
        return self.actions_encodings_list[index]

    def __len__(self):
        return len(self.__actions)
