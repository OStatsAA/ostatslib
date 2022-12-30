# pylint: disable=unused-argument
"""
test_calculate_score_reward function tests module
"""

import math
from ostatslib.actions.utils import calculate_score_reward


def test_reward_for_nan_scores() -> None:
    """
    Tests reward is negative when scores are NaN
    """
    reward = calculate_score_reward(math.nan)
    assert reward < 0
