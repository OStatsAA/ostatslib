# pylint: disable=redefined-outer-name
"""
Epsilon-greedy testing module
"""

import string
from collections import Counter
from unittest.mock import Mock
from scipy.stats import kstest, uniform
import pytest

from ostatslib.exploration_strategies import EpsilonGreedy
from ostatslib.features_extractors import AnalysisFeaturesSet, DataFeaturesSet
from ostatslib.states import State

BEST_ACTION = 'best_action'


@pytest.fixture
def dummy_state() -> State:
    """
    Instantiates a dummy state fixture
    """
    return State(DataFeaturesSet(), AnalysisFeaturesSet())


@pytest.fixture
def model_mock() -> Mock:
    """
    Reinforcement Learning model mock
    """
    mock = Mock()
    attrs = {
        'is_fit.return_value': False,
        'fit.return_value': None,
        'predict.return_value': BEST_ACTION
    }
    mock.configure_mock(**attrs)
    return mock


def test_actions_taken_uniformly_if_epsilon_is_1(model_mock: Mock,
                                                 dummy_state: State) -> None:
    """
    If epsilon = 1, the epsilon-greedy strategy picks actions uniformly with prob 1/n_actions.
    Kolmogorov-Smirnov test for goodness of fit used to test uniformity
    """
    strategy = EpsilonGreedy(1)
    actions_list = list(string.ascii_lowercase[:9]) + [BEST_ACTION]
    explored_actions = [""] * 1000

    for i, _ in enumerate(explored_actions):
        explored_actions[i] = strategy.get_action(
            model_mock, dummy_state, actions_list, dummy_state)

    actions_count = list(Counter(explored_actions).values())
    param_a = min(actions_count)
    param_b = max(actions_count)
    kstest_result = kstest(actions_count, uniform(param_a, param_b).cdf)
    assert kstest_result.pvalue < .01


def test_best_action_taken_aprox_half_times_if_epsilon_is_half(model_mock: Mock,
                                                               dummy_state: State) -> None:
    """
    If epsilon = 0.5,
    the epsilon-greedy strategy picks the best action approximately 50% of the times.
    """
    strategy = EpsilonGreedy(.5)
    actions_list = list(string.ascii_lowercase[:9]) + [BEST_ACTION]
    explored_actions = [""] * 1000

    for i, _ in enumerate(explored_actions):
        explored_actions[i] = strategy.get_action(
            model_mock, dummy_state, actions_list, dummy_state)

    actions_counter = Counter(explored_actions)
    most_freq_action, most_freq_value = actions_counter.most_common(1)[0]
    assert most_freq_action == BEST_ACTION
    assert 400 < most_freq_value < 600


def test_best_action_is_always_taken_if_epsilon_is_0(model_mock: Mock,
                                                     dummy_state: State) -> None:
    """
    If epsilon = 1, the epsilon-greedy strategy picks actions uniformly with prob 1/n_actions.
    Kolmogorov-Smirnov test for goodness of fit used to test uniformty
    """
    strategy = EpsilonGreedy(0)
    actions_list = list(string.ascii_lowercase[:9]) + [BEST_ACTION]
    explored_actions = [""] * 1000

    for i, _ in enumerate(explored_actions):
        explored_actions[i] = strategy.get_action(
            model_mock, dummy_state, actions_list, dummy_state)

    actions_counter = Counter(explored_actions)
    most_freq_action, most_freq_value = actions_counter.most_common(1)[0]
    assert most_freq_action == BEST_ACTION
    assert most_freq_value == 1000
