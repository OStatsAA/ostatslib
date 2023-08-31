"""
OstatsLib Configuration module
"""

from typing import TypedDict


class Config(TypedDict):
    """
    Configuration TypedDict
    """
    # Diagnostics tests
    FULL_PENALIZED_PVALUE: float
    PARTIAL_PENALIZED_PVALUE: float

    # Environment
    MAX_STEPS: int

    # Rewards limits
    MIN_REWARD: float
    MAX_REWARD: float
    EXPLORATORY_REWARD_FRACTION: float
    MAX_EXPLORATORY_REWARD: float

    # Scores
    MIN_ACCEPTED_SCORE: float

    # Runtime
    FIT_TIMEOUT: int


DEFAULT_CONFIG = Config(
    FULL_PENALIZED_PVALUE=0.01,
    PARTIAL_PENALIZED_PVALUE=0.05,
    MAX_STEPS=15,
    MIN_REWARD=-1,
    MAX_REWARD=1,
    EXPLORATORY_REWARD_FRACTION=0.1,
    MAX_EXPLORATORY_REWARD=0.1,
    MIN_ACCEPTED_SCORE=0.7,
    FIT_TIMEOUT=600
)
