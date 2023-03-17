# pylint: disable=unused-argument
"""
test_update_state_score function tests module
"""

import math
from ostatslib.actions.utils import update_state_score
from ostatslib.states import State


def test_for_nan_scores() -> None:
    """
    State score should be 0.
    Must ensure state score is a number to prevent NaN propagation
    """
    state = State()
    update_state_score(state, math.nan)
    assert state.get('score') == 0
