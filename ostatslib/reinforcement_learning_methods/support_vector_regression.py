"""
SupportVectorRegression module
"""

from random import choice
import numpy as np
from pandas import DataFrame
from sklearn.svm import SVR

from ostatslib.environments import Environment
from ostatslib.exploration_strategies import ExplorationStrategy, EpsilonGreedy
from ostatslib.states import State

from .reinforcement_learning_method import ReinforcementLearningMethod
from .utils import ModelNotFitError


class SupportVectorRegression(ReinforcementLearningMethod):
    """
    Regression model for agents predictions using SVR
    """

    def __init__(self,
                 svr: SVR = None,
                 exploratory_strategy: ExplorationStrategy = None) -> None:
        self.__svr = svr if svr is not None else SVR()
        if exploratory_strategy is None:
            self.__exploratory_strategy = EpsilonGreedy()
        self.__is_fit: bool = False
        self.__states_features_history = []
        self.__actions_history = []
        self.__rewards_history = []

    @property
    def is_fit(self) -> bool:
        return self.__is_fit

    def run_analysis(self,
                     state: State,
                     data: DataFrame,
                     environment: Environment,
                     max_steps: int) -> list:
        if not self.__is_fit:
            raise ModelNotFitError()

        steps = []
        for _ in range(1, max_steps):
            action_result, done = self.__run_analysis_step(data,
                                                           environment,
                                                           state)
            state = action_result.state
            steps.append(action_result)

            if done:
                break

        return steps

    def run_training_episode(self,
                             state: State,
                             data: DataFrame,
                             environment: Environment,
                             max_steps: int) -> float:
        episode_reward = 0

        for _ in range(1, max_steps):
            action_result, done = self.__run_training_step(data,
                                                           environment,
                                                           state)

            state = action_result.state
            episode_reward += action_result.reward

            if done:
                break

        self.__fit()
        self.__is_fit = True
        return episode_reward

    def __run_analysis_step(self,
                            data: DataFrame,
                            environment: Environment,
                            state: State):
        available_actions = environment.actions_space.actions_encodings_list
        actions_values = self.__predict(state.features_vector,
                                        available_actions)
        action = available_actions[np.argmax(actions_values)]
        action_result, done = environment.run_action(state, data, action)

        return action_result, done

    def __run_training_step(self,
                            data: DataFrame,
                            environment: Environment,
                            state: State):
        action = None
        available_actions = environment.actions_space.actions_encodings_list
        if self.__is_fit:
            action = self.__exploratory_strategy.get_action(state,
                                                            available_actions,
                                                            self.__predict)
        else:
            action = choice(available_actions)

        # Apply the sampled action in our environment
        action_result, done = environment.run_action(state,
                                                     data,
                                                     action)
        self.__states_features_history.append(state.features_vector)
        self.__actions_history.append(action)
        self.__rewards_history.append(action_result.reward)
        return action_result, done

    def __fit(self) -> None:
        features = np.concatenate((self.__states_features_history,
                                   self.__actions_history),
                                  axis=1)
        self.__svr.fit(features, self.__rewards_history)
        self.__is_fit = True

    def __predict(self,
                  state_features: np.ndarray,
                  available_actions: np.ndarray) -> np.ndarray:
        features = None
        if state_features.ndim == available_actions.ndim == 1:
            features = np.concatenate(
                (state_features, available_actions)).reshape(1, -1)
        elif state_features.ndim == available_actions.ndim:
            features = np.concatenate(
                (state_features, available_actions), axis=1)
        else:
            state_features = np.tile(
                state_features, (len(available_actions), 1))
            features = np.concatenate(
                (state_features, available_actions), axis=1)

        predictions = self.__svr.predict(features)

        return predictions
