# pylint: disable=unused-argument
"""
explainability rewards decorators functions tests module
"""

from random import uniform

from ostatslib.actions.utils import (ActionResult,
                                     comprehensible_model,
                                     interpretable_model,
                                     opaque_model,
                                     REWARD_LOWER_LIMIT,
                                     REWARD_UPPER_LIMIT)
from ostatslib.actions.utils.explainability_rewards import (COMPREHENSIBLE_REWARD,
                                                            INTERPETRABLE_REWARD,
                                                            OPAQUE_PENALTY)


def test_opaque_model() -> None:
    """
    Tests if opaque models penalty is applied to action_fn
    """
    random_reward = uniform(REWARD_LOWER_LIMIT, REWARD_UPPER_LIMIT)

    @opaque_model
    def action_fn(*args):
        return ActionResult(None, random_reward, None)

    action_result = action_fn(None, None)
    assert action_result.reward == (random_reward + OPAQUE_PENALTY)


def test_interpetrable_model() -> None:
    """
    Tests if interpetrable models reward is applied to action_fn
    """
    random_reward = uniform(REWARD_LOWER_LIMIT, REWARD_UPPER_LIMIT)

    @interpretable_model
    def action_fn(*args):
        return ActionResult(None, random_reward, None)

    action_result = action_fn(None, None)
    assert action_result.reward == (random_reward + INTERPETRABLE_REWARD)


def test_comprehensible_model() -> None:
    """
    Tests if comprehensible models reward is applied to action_fn
    """
    random_reward = uniform(REWARD_LOWER_LIMIT, REWARD_UPPER_LIMIT)

    @comprehensible_model
    def action_fn(*args):
        return ActionResult(None, random_reward, None)

    action_result = action_fn(None, None)
    assert action_result.reward == (random_reward + COMPREHENSIBLE_REWARD)
