"""
DataFeaturesSets module
"""

from dataclasses import dataclass, field
from gymnasium.spaces import Box

from ostatslib.states.features_set import FeaturesSet


@dataclass(init=False)
class DataFeaturesSet(FeaturesSet):
    """
    Class to hold features extracted from a dataset.
    """
    log_rows_count: float = field(
        default=0,
        metadata={
            'gym_space': Box(0, 1),
            'get_value_fn': None
        })

    is_response_dichotomous: int = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    is_response_quantitative: int = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    is_response_discrete: int = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    is_response_positive_values_only: int = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })
