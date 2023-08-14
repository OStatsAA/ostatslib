from abc import ABC, abstractmethod
from inspect import signature
import math
from typing import Any, Callable, Generic, TypeVar
from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.base import BaseEstimator
from sklearn.svm import SVC

from ostatslib.config import Config
from ostatslib.states import State
from .utils import split_x_y_data

T = TypeVar("T", BaseEstimator, SVC)


class ActionInfo(dict, Generic[T]):
    action_name: str
    model: None | T
    raised_exception: bool
    is_invalid_state: bool
    next_state: None | State

    def __init__(self,
                 action_name: str,
                 model: None | T = None,
                 raised_exception: bool = False,
                 is_invalid_state: bool = False,
                 next_state: None | State = None) -> None:
        self.action_name = action_name
        self.model = model
        self.raised_exception = raised_exception
        self.is_invalid_state = is_invalid_state
        self.next_state = next_state
        super(ActionInfo, self).__init__()


class Action(ABC):

    action_name: str
    action_key: str
    exceptions_handlers: dict[Exception,
                              Callable[[Exception, State, Config], None]] | None = None
    validations: list[tuple[str, Callable[..., bool], Any]] | None = None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Action:
            return any([action_type == cls for action_type in C.__mro__])

        return False

    def _exception_handler(self, error: Exception, state: State, config: Config) -> None:
        """Handles exceptions on model fitting.

        Args:
            error (Exception): exception
            state (State): current state

        Returns:
            None: None
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
    def execute(self, data: DataFrame, state: State, config: Config) -> tuple[State, float, ActionInfo]:
        """Executes and actions

        Args:
            data (DataFrame): data
            state (State): current state
            config (Config): configuration dict

        Returns:
            State, float: next state, reward
        """


class ExploratoryAction(Action):

    def _calculate_reward(self, state: State, state_copy: State, config: Config) -> float:
        if state == state_copy:
            return config['MIN_REWARD']

        return config['MAX_EXPLORATORY_REWARD']

    def _update_state(self, state: State, value: str | int | bool | float) -> State:
        state.set(self.action_key, value)
        return state

    @abstractmethod
    def _explore(self, data: DataFrame, state: State) -> str | int | bool | float:
        ...

    def execute(self, data: DataFrame, state: State, config: Config) -> tuple[State, float, ActionInfo]:
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

        reward = self._calculate_reward(state, state_copy, config)
        state = self._update_state(state, exploratory_value)
        info.next_state = state.copy()
        return state, reward, info


class TargetExploratoryAction(ExploratoryAction):

    def _validate_state(self, state: State) -> bool:
        if bool(state.get('response_variable_label')):
            return super()._validate_state(state)

        return False


class ModelEstimatorAction(Action, Generic[T]):

    estimator: T
    params_grid: dict | None = None

    def _exception_handler(self, error: Exception, state: State, config: Config) -> None:
        state.set(self.action_key + "_score_reward", config['MIN_REWARD'])
        return super()._exception_handler(error, state, config)

    def _calculate_reward(self, score: float, config: Config) -> float:
        """Calculates action reward

        Args:
            state (State): current state
            score (float): score

        Returns:
            float: reward
        """
        if math.isnan(score) or (not -1 <= score <= 1):
            return -1

        if score <= config['MIN_ACCEPTED_SCORE']:
            return - (1 - score)

        return score

    def _update_state(self, state: State, reward: float, score: float) -> State:
        """Updates state before completing action execution

        Args:
            state (State): state

        Returns:
            State: next state
        """
        state.set(self.action_key + '_score_reward', reward)

        if math.isnan(score) or (not -1 <= score <= 1):
            state.set('score', 0)
        else:
            state.set('score', score)

        return state

    @abstractmethod
    def _fit(self, data: DataFrame, state: State) -> tuple[T, float]:
        """Private model fitting method

        Args:
            x_data (DataFrame): data
            y_data (Series): response values
        """

    def execute(self, data: DataFrame, state: State, config: Config) -> tuple[State, float, ActionInfo[T]]:
        info = ActionInfo[T](self.action_name)
        if not self._validate_state(state):
            info.is_invalid_state = True
            return state, config['MIN_REWARD'], info

        model: T
        score: float
        try:
            model, score = self._fit(data, state)
            info.model = model
        except Exception as error:
            self._exception_handler(error, state, config)
            info.raised_exception = True
            info.next_state = state.copy()
            return state, config['MIN_REWARD'], info

        reward = self._calculate_reward(score, config)
        state = self._update_state(state, reward, score)
        info.next_state = state.copy()
        return state, reward, info


class TargetModelEstimatorAction(ModelEstimatorAction[T]):

    def _validate_state(self, state: State) -> bool:
        if bool(state.get('response_variable_label')):
            return super()._validate_state(state)

        return False

    def _fit(self, data: DataFrame, state: State) -> tuple[T, float]:
        x_data, y_data = split_x_y_data(data, state)

        if self.params_grid is None:
            cv_search: dict[str, ndarray] = cross_validate(self.estimator,
                                                           x_data,
                                                           y_data,
                                                           return_estimator=True)
            best_index = cv_search['test_score'].argmax()
            best_estimator: T = cv_search['estimator'][best_index]
            best_score: float = cv_search['test_score'][best_index]
            return best_estimator, best_score

        search = GridSearchCV(self.estimator,
                              self.params_grid,
                              n_jobs=-1).fit(x_data, y_data)
        return search.best_estimator_, search.best_score_


MIN_TREE_DEPTH = 2
MAX_TREE_DEPTH = 20


class TreeEstimatorAction(TargetModelEstimatorAction[T]):

    def _fit(self, data: DataFrame, state: State) -> tuple[T, float]:
        if self.params_grid is not None:
            max_depth = len(data.columns) // 2
            self.params_grid['max_depth'] = min(max(max_depth, MIN_TREE_DEPTH),
                                                MAX_TREE_DEPTH)
        return super()._fit(data, state)
