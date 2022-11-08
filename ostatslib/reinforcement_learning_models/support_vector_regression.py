"""
SupportVectorRegression module
"""

from numpy import argmax, inf, ndarray, concatenate, tile
from sklearn.svm import SVR
from ostatslib.reinforcement_learning_models.model import Model


class SupportVectorRegression(Model):
    """
    Regression model for agents predictions using SVR
    """

    def __init__(self, svr = None) -> None:
        self.__svr = svr if svr is not None else SVR()
        self.__is_fit: bool = False

    @property
    def is_fit(self) -> bool:
        return self.__is_fit

    def fit(self,
            state_features: ndarray,
            actions_features: ndarray,
            expected_rewards: ndarray) -> None:
        features = concatenate((state_features, actions_features), axis=1)
        self.__svr.fit(features, expected_rewards)
        self.__is_fit = True

    def predict(self,
                state_features: ndarray,
                available_actions: ndarray) -> ndarray:
        if len(state_features.shape) == 1:
            state_features = tile(state_features, (len(available_actions), 1))

        features = concatenate((state_features, available_actions), axis=1)
        predictions = self.__svr.predict(features)

        return available_actions[argmax(predictions)]
