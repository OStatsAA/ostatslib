# pylint: disable=unused-argument
"""
reward_cap decorator function tests module
"""

from ostatslib.actions.utils import (ActionResult,
                                     reward_cap,
                                     REWARD_LOWER_LIMIT,
                                     REWARD_UPPER_LIMIT)


def test_enforces_reward_lower_limit() -> None:
    """
    Tests if the reward_cap decorator ensures lower limit reward value
    """
    @reward_cap
    def too_low_reward_action(*args):
        return ActionResult(None, REWARD_LOWER_LIMIT - 1, None)

    action_result = too_low_reward_action(None, None)
    assert action_result.reward == REWARD_LOWER_LIMIT


def test_enforces_reward_upper_limit() -> None:
    """
    Tests if the reward_cap decorator ensures upper limit reward value
    """
    @reward_cap
    def too_high_reward_action(*args):
        return ActionResult(None, REWARD_UPPER_LIMIT + 1, None)

    action_result = too_high_reward_action(None, None)
    assert action_result.reward == REWARD_UPPER_LIMIT


def test_do_not_change_reward_if_it_is_within_limits() -> None:
    """
    Tests wether the reward_cap decorator does not change the action rewards if \
        it's within limits
    """
    reward = (REWARD_LOWER_LIMIT + REWARD_UPPER_LIMIT)/2

    @reward_cap
    def within_limits_reward_action(*args):
        return ActionResult(None, reward, None)

    action_result = within_limits_reward_action(None, None)
    assert action_result.reward == reward
