# pylint: disable=unused-argument
"""
reward_cap decorator function tests module
"""

from pandas import DataFrame
from ostatslib import config
from ostatslib.actions import ActionInfo
from ostatslib.actions.utils import reward_cap
from ostatslib.states import State


def test_enforces_reward_lower_limit() -> None:
    """
    Tests if the reward_cap decorator ensures lower limit reward value
    """
    @reward_cap
    def too_low_reward_action(*args):
        return State(), config.MIN_REWARD - 1, ActionInfo(action_name="Test",
                                                          action_fn=too_low_reward_action,
                                                          model=None,
                                                          raised_exception=False)

    reward = too_low_reward_action(State(), DataFrame())[1]
    assert reward == config.MIN_REWARD


def test_enforces_reward_upper_limit() -> None:
    """
    Tests if the reward_cap decorator ensures upper limit reward value
    """
    @reward_cap
    def too_high_reward_action(*args):
        return State(), config.MAX_REWARD + 1, ActionInfo(action_name="Test",
                                                          action_fn=too_high_reward_action,
                                                          model=None,
                                                          raised_exception=False)

    reward = too_high_reward_action(State(), DataFrame())[1]
    assert reward == config.MAX_REWARD


def test_do_not_change_reward_if_it_is_within_limits() -> None:
    """
    Tests wether the reward_cap decorator does not change the action rewards if \
        it's within limits
    """
    @reward_cap
    def within_limits_reward_action(*args):
        _reward = (config.MIN_REWARD + config.MAX_REWARD)/2
        return State(), _reward, ActionInfo(action_name="Test",
                                            action_fn=within_limits_reward_action,
                                            model=None,
                                            raised_exception=False)

    reward = within_limits_reward_action(State(), DataFrame())[1]
    assert config.MIN_REWARD <= reward <= config.MAX_REWARD
