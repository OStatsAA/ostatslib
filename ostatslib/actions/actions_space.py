"""
ActionsSpace module
"""

from functools import cached_property
from gymnasium.spaces import MultiBinary

import numpy as np
import numpy.typing as npt
from pandas import DataFrame

from ostatslib.actions.exploratory_actions import (
    get_log_rows_count,
    infer_response_dtype,
    is_response_dichotomous_check,
    is_response_discrete_check,
    is_response_positive_values_only_check,
    is_response_quantitative_check,
    time_convertible_variable_search
)
from ostatslib.actions.regression_models import (
    linear_regression,
    poisson_regression,
    support_vector_regression,
    decision_tree_regression,
    time_series_auto_arima
)
from ostatslib.actions.classifiers import (
    logistic_regression,
    support_vector_classification,
    decision_tree
)
from ostatslib.actions.clustering import (
    k_means,
    dbscan
)
from ostatslib.actions.utils import as_binary_array
from ostatslib.states import State
from .action import Action, ActionInfo, ActionResult


MaskNDArray = npt.NDArray[np.int8]
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
                                       as_binary_array(4, ENCODING_LENGTH)),
    'time_convertible_variable_search': (time_convertible_variable_search,
                                         as_binary_array(5, ENCODING_LENGTH)),
    'infer_response_dtype': (infer_response_dtype,
                             as_binary_array(6, ENCODING_LENGTH))
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
                                  as_binary_array(18, ENCODING_LENGTH)),
    'decision_tree_regression': (decision_tree_regression,
                                 as_binary_array(19, ENCODING_LENGTH)),
    'time_series_auto_arima': (time_series_auto_arima,
                               as_binary_array(20, ENCODING_LENGTH))
}

# Encoding: 24 to 31
CLUSTERING = {
    'k_means': (k_means,
                as_binary_array(24, ENCODING_LENGTH)),
    'dbscan': (dbscan,
               as_binary_array(25, ENCODING_LENGTH))
}


def _invalid_action_step(state: State, data: DataFrame) -> ActionResult[None]:
    reward = float(-1)
    info = ActionInfo(action_name='Invalid Action',
                      action_fn=_invalid_action_step,
                      model=None,
                      raised_exception=False)
    return state, reward, info


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
