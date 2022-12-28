"""
calculate score reward helper function module
"""

import math


MIN_SCORE_VALUE: float = 0.6


def calculate_score_reward(score: float) -> float:
    """
    Calculates reward based on standard score.

    Args:
        score (float): standard score

    Returns:
        float: reward
    """
    if math.isnan(score):
        return -1

    if score <= MIN_SCORE_VALUE:
        return - (1 - score)

    return score
