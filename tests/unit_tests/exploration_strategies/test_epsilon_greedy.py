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
from ostatslib.states import State

BEST_ACTION = 'best_action'
ACTIONS_LIST = list(string.ascii_lowercase[:9]) + [BEST_ACTION]


@pytest.fixture
def predict_fn_mock() -> Mock:
    """
    Reinforcement Learning model mock
    """
    mock = Mock()
    predictions = [0] * 10
    predictions[ACTIONS_LIST.index(BEST_ACTION)] = 1
    attrs = {
        'return_value': predictions
    }
    mock.configure_mock(**attrs)
    return mock


def test_actions_taken_uniformly_if_epsilon_is_1(predict_fn_mock: Mock) -> None:
    """
    If epsilon = 1, the epsilon-greedy strategy picks actions uniformly with prob 1/n_actions.
    Kolmogorov-Smirnov test for goodness of fit used to test uniformity
    """
    strategy = EpsilonGreedy(1)
    explored_actions = [""] * 1000

    for i, _ in enumerate(explored_actions):
        explored_actions[i] = strategy.get_action(State(),
                                                  ACTIONS_LIST,
                                                  predict_fn_mock)

    actions_count = list(Counter(explored_actions).values())
    param_a = min(actions_count)
    param_b = max(actions_count)
    kstest_result = kstest(actions_count, uniform(param_a, param_b).cdf)
    assert kstest_result.pvalue < .01


def test_best_action_taken_aprox_half_times_if_epsilon_is_half(predict_fn_mock: Mock) -> None:
    """
    If epsilon = 0.5,
    the epsilon-greedy strategy picks the best action approximately 50% of the times.
    """
    strategy = EpsilonGreedy(.5)
    explored_actions = [""] * 1000

    for i, _ in enumerate(explored_actions):
        explored_actions[i] = strategy.get_action(State(),
                                                  ACTIONS_LIST,
                                                  predict_fn_mock)

    actions_counter = Counter(explored_actions)
    most_freq_action, most_freq_value = actions_counter.most_common(1)[0]
    assert most_freq_action == BEST_ACTION
    assert 400 < most_freq_value < 600


def test_best_action_is_always_taken_if_epsilon_is_0(predict_fn_mock: Mock) -> None:
    """
    If epsilon = 1, the epsilon-greedy strategy picks actions uniformly with prob 1/n_actions.
    Kolmogorov-Smirnov test for goodness of fit used to test uniformty
    """
    strategy = EpsilonGreedy(0)
    explored_actions = [""] * 1000

    for i, _ in enumerate(explored_actions):
        explored_actions[i] = strategy.get_action(State(),
                                                  ACTIONS_LIST,
                                                  predict_fn_mock)

    actions_counter = Counter(explored_actions)
    most_freq_action, most_freq_value = actions_counter.most_common(1)[0]
    assert most_freq_action == BEST_ACTION
    assert most_freq_value == 1000
