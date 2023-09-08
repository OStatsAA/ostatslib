"""
Actions base classes module
"""

from abc import ABC, abstractmethod
from inspect import signature
import math
import time
from typing import Any, Callable, Generic, TypeVar
import warnings
from numpy import ndarray
from pandas import DataFrame
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from statsforecast.models import AutoARIMA
from wrapt_timeout_decorator.wrapt_timeout_decorator import timeout

from ostatslib.config import Config
from ostatslib.states import State
from .utils import split_x_y_data

T = TypeVar("T", BaseEstimator, SVC, AutoARIMA)


class ActionInfo(dict, Generic[T]):
    """
    Information class populated by action or environment during a step.
    Ostatslib implements Gymnasium step returns
    `ref <https://gymnasium.farama.org/api/env/#gymnasium.Env.step>`
    """
    action_name: str
    model: None | T
    raised_exception: bool
    is_invalid_state: bool
    next_state: None | State
    fit_time: None | float

    def __init__(self,
                 action_name: str,
                 model: None | T = None,
                 raised_exception: bool = False,
                 is_invalid_state: bool = False,
                 next_state: None | State = None,
                 fit_time: None | float = None) -> None:
        self.action_name = action_name
        self.model = model
        self.raised_exception = raised_exception
        self.is_invalid_state = is_invalid_state
        self.next_state = next_state
        self.fit_time = fit_time
        super().__init__()


class Action(ABC):
    """
    Action base abstract class
    """

    action_name: str
    action_key: str
    exceptions_handlers: dict[Exception,
                              Callable[[Exception, State, Config], None]] | None = None
    validations: list[tuple[str, Callable[..., bool], Any]] | None = None

    @classmethod
    def __subclasshook__(cls, class_: type):
        if cls is Action:
            return any(action_type for action_type in class_.__mro__ if action_type == cls)

        return False

    def _exception_handler(self, error: Exception, state: State, config: Config) -> None:
        """Handles exceptions on model fitting.

        Args:
            error (Exception): thrown exception in fitting/exploring
            state (State): state
            config (Config): configuration dictionary

        Returns:
            None: return None
        """
        if self.exceptions_handlers is None:
            return None

        handler = self.exceptions_handlers.get(error)
        if handler is None:
            return None

        return handler(error, state, config)

    def _validate_state(self, state: State) -> bool:
        """Private validation method. Iterates over validations and if any fails, return False.

        Args:
            state (State): current state

        Returns:
            bool: true if state is valid
        """
        if self.validations is None:
            return True

        for (feature_key, operator_fn, value) in self.validations:
            is_valid = False
            if value is None:
                operator_fn_signature = signature(operator_fn)
                if len(operator_fn_signature.parameters) == 1:
                    is_valid = operator_fn(state.get(feature_key))
            else:
                is_valid = operator_fn(state.get(feature_key), value)

            if not is_valid:
                return False

        return True

    @abstractmethod
    def execute(self,
                data: DataFrame,
                state: State,
                config: Config) -> tuple[State, float, ActionInfo]:
        """Executes action

        Args:
            data (DataFrame): data
            state (State): state
            config (Config): configuration dictionary

        Returns:
            tuple[State, float, ActionInfo]: tuple of next state, reward and action info
        """


class ExploratoryAction(Action):
    """Extends action base class for exploratory actions
    """

    def _calculate_reward(self, state: State, state_copy: State, config: Config) -> float:
        """Calculates exploratory reward based on configuration dictionary settings

        Args:
            state (State): state
            state_copy (State): copy holding initial state values
            config (Config): configuration dictionary

        Returns:
            float: reward
        """
        if state == state_copy:
            return config['MIN_REWARD']

        return config['MAX_EXPLORATORY_REWARD']

    def _update_state(self, state: State, value: str | int | bool | float) -> State:
        """Updates state using action_key property and exploration value

        Args:
            state (State): state
            value (str | int | bool | float): feature value

        Returns:
            State: updated state
        """
        state.set(self.action_key, value)
        return state

    @abstractmethod
    def _explore(self, data: DataFrame, state: State) -> str | int | bool | float:
        """Private explore method called by public execute method

        Args:
            data (DataFrame): data
            state (State): state

        Returns:
            str | int | bool | float: exploratory value
        """

    def execute(self,
                data: DataFrame,
                state: State,
                config: Config) -> tuple[State, float, ActionInfo]:
        info = ActionInfo(self.action_name)
        if not self._validate_state(state):
            info.is_invalid_state = True
            return state, config['MIN_REWARD'], info

        state_copy = state.copy()
        try:
            exploratory_value = self._explore(data, state)
        except Exception as error:
            self._exception_handler(error, state, config)
            info.raised_exception = True
            info.next_state = state.copy()
            return state, config['MIN_REWARD'], info

        state = self._update_state(state, exploratory_value)
        reward = self._calculate_reward(state, state_copy, config)
        info.next_state = state.copy()
        return state, reward, info


class TargetExploratoryAction(ExploratoryAction):
    """Extends ExploraotryAction base class for
    exploratory actions that inspect response variable values
    """

    def _validate_state(self, state: State) -> bool:
        if bool(state.get('response_variable_label')):
            return super()._validate_state(state)

        return False


class ModelEstimatorAction(Action, Generic[T]):
    """Extends base action class for model estimators
    """

    estimator: T
    params_grid: dict | None = None

    def _exception_handler(self, error: Exception, state: State, config: Config) -> None:
        state.set(self.action_key + "_score_reward", config['MIN_REWARD'])
        return super()._exception_handler(error, state, config)

    def _calculate_reward(self, score: float, config: Config) -> float:
        """Calculates reward based on model score and configuration dictionary

        Args:
            score (float): score value
            config (Config): configuration dictionary

        Returns:
            float: reward
        """
        if math.isnan(score) or (not 0 <= score <= 1):
            return config['MIN_REWARD']

        if score < config['MIN_ACCEPTED_SCORE']:
            return - (1 - score)

        return score

    def _update_state(self, state: State, reward: float, score: float) -> State:
        """Updates state on action key based on reward and score

        Args:
            state (State): state
            reward (float): reward
            score (float): model score

        Returns:
            State: updated state
        """
        state.set(self.action_key + '_score_reward', reward)

        if math.isnan(score) or (not -1 <= score <= 1):
            state.set('score', 0)
        else:
            state.set('score', score)

        return state

    @abstractmethod
    def _fit(self, data: DataFrame, state: State, config: Config) -> tuple[T, float]:
        """Private fit method

        Args:
            data (DataFrame): data
            state (State): state
            config (Config): configuration dictionary

        Returns:
            tuple[T, float]: adjusted model and score
        """

    def execute(self,
                data: DataFrame,
                state: State,
                config: Config) -> tuple[State, float, ActionInfo[T]]:
        info = ActionInfo[T](self.action_name)
        if not self._validate_state(state):
            info.is_invalid_state = True
            return state, config['MIN_REWARD'], info

        model: T
        score: float
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=ConvergenceWarning)
            fit_start = time.perf_counter()
            try:
                model, score = self._fit(data, state, config)
                info.fit_time = time.perf_counter() - fit_start
                info.model = model
            except Exception as error:
                if isinstance(error, TimeoutError):
                    print(error.args[0])
                info.fit_time = time.perf_counter() - fit_start
                self._exception_handler(error, state, config)
                info.raised_exception = True
                info.next_state = state.copy()
                return state, config['MIN_REWARD'], info

        reward = self._calculate_reward(score, config)
        state = self._update_state(state, reward, score)
        info.next_state = state.copy()
        return state, reward, info


class TargetModelEstimatorAction(ModelEstimatorAction[T]):
    """Extends ModelEstimatorAction for modeling response (target) variable
    """

    def _validate_state(self, state: State) -> bool:
        if bool(state.get('response_variable_label')):
            return super()._validate_state(state)

        return False

    def _fit(self, data: DataFrame, state: State, config: Config) -> tuple[T, float]:
        x_data, y_data = split_x_y_data(data, state)
        _timeout = timeout(config['FIT_TIMEOUT'])

        if self.params_grid is None:
            cv_search: dict[str, ndarray] = _timeout(cross_validate)(self.estimator,
                                                                     x_data,
                                                                     y_data,
                                                                     return_estimator=True,
                                                                     n_jobs=4,
                                                                     verbose=config['FIT_VERBOSE'])
            best_index = cv_search['test_score'].argmax()
            best_estimator: T = cv_search['estimator'][best_index]
            best_score: float = cv_search['test_score'][best_index]
            return best_estimator, best_score

        search = GridSearchCV(self.estimator,
                              self.params_grid,
                              n_jobs=4,
                              verbose=config['FIT_VERBOSE'])
        search = _timeout(search.fit)(x_data, y_data)
        return search.best_estimator_, search.best_score_


class TreeEstimatorAction(TargetModelEstimatorAction[T]):
    """Extends TargetModelEstimatorAction for Tree models
    """
    _MIN_TREE_DEPTH = 2
    _MAX_TREE_DEPTH = 10

    def _fit(self, data: DataFrame, state: State, config: Config) -> tuple[T, float]:
        if self.params_grid is not None:
            max_depth = len(data.columns) // 2
            self.params_grid['max_depth'] = [min(max(max_depth, self._MIN_TREE_DEPTH),
                                                 self._MAX_TREE_DEPTH)]
        return super()._fit(data, state, config)
