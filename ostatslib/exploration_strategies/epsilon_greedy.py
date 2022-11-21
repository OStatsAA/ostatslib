"""
EpsilonGreedy class module
"""

from random import random, choice

from numpy import argmax, ndarray
from ostatslib.states import State
from .exploration_strategy import ExplorationStrategy


class EpsilonGreedy(ExplorationStrategy):
    """
    Class implementing the epsilon-greedy exploration strategy.\n
    Epsilon = 1 -> randomly selects actions;\n
    Epsilon = 0 -> always selects estimated best action
    """

    def __init__(self, epsilon: float = .9) -> None:
        self.__epsilon = epsilon

    def get_action(self,
                   state: State,
                   actions: ndarray,
                   predict_fn: callable) -> ndarray:
        prob = random()
        if prob > self.__epsilon:
            predictions = predict_fn(state.features_vector, actions)
            return actions[argmax(predictions)]

        return choice(actions)
