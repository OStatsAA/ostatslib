"""
Model abstract class module
"""

from abc import ABC, abstractmethod
from numpy import ndarray


class Model(ABC):
    """
    Model abstract class
    """

    @abstractmethod
    def fit(self,
            state_features: ndarray,
            actions_features: ndarray,
            expected_rewards: ndarray) -> None:
        """
        Model fitting method

        Args:
            state_features (ndarray): state features vector
            actions_features (ndarray): actions features
            expected_rewards (ndarray): rewards array
        """

    @abstractmethod
    def predict(self,
                state_features: ndarray,
                available_actions: ndarray) -> ndarray:
        """
        Model predict method

        Args:
            features (ndarray): state features vector
            actions_features (ndarray): actions features

        Returns:
            ndarray: predicted best action code
        """
