"""
update state score helper function module
"""

import math

from ostatslib.states import State


def update_state_score(state: State, score: float) -> State:
    """
    Updates State score attribute

    Args:
        state (State): state
        score (float): action score

    Returns:
        State: state
    """
    if math.isnan(score):
        return state

    state.set('score', score)
    return state
