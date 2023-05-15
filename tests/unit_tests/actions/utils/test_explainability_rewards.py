# pylint: disable=unused-argument
"""
explainability rewards decorators functions tests module
"""

from random import uniform

from pandas import DataFrame
from ostatslib.actions import ActionInfo

from ostatslib import config
from ostatslib.actions.utils import (comprehensible_model,
                                     interpretable_model,
                                     opaque_model)
from ostatslib.actions.utils.explainability_rewards import (COMPREHENSIBLE_REWARD,
                                                            INTERPRETABLE_REWARD,
                                                            OPAQUE_PENALTY)
from ostatslib.states import State

random_reward = uniform(config.MIN_REWARD, config.MAX_REWARD)


def __action_fn(*args):
    return State(), random_reward, ActionInfo(action_name="Test",
                                                action_fn=__action_fn,
                                                model=None,
                                                raised_exception=False)


def test_opaque_model() -> None:
    """
    Tests if opaque models penalty is applied to action_fn
    """
    @opaque_model
    def action_fn(*args):
        return __action_fn()

    reward = action_fn(State(), DataFrame())[1]
    assert reward == (random_reward + OPAQUE_PENALTY)


def test_interpetrable_model() -> None:
    """
    Tests if interpretable models reward is applied to action_fn
    """
    @interpretable_model
    def action_fn(*args):
        return __action_fn()

    reward = action_fn(State(), DataFrame())[1]
    assert reward == (random_reward + INTERPRETABLE_REWARD)


def test_comprehensible_model() -> None:
    """
    Tests if comprehensible models reward is applied to action_fn
    """
    @comprehensible_model
    def action_fn(*args):
        return __action_fn()

    reward = action_fn(State(), DataFrame())[1]
    assert reward == (random_reward + COMPREHENSIBLE_REWARD)
