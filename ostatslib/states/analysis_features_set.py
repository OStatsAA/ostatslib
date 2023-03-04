"""
AnalysisFeaturesSet module
"""

from dataclasses import dataclass, field
from gymnasium.spaces import Discrete, Box

from ostatslib.states.features_set import FeaturesSet


def time_convertable_variable_to_feature(time_convertable_variable: str) -> bool:
    """
    Returns -1 if field is set to None, else returns boolean from string.
    Empty string = False and any valid string = True

    Args:
        time_convertable_variable (str): analysis features set field

    Returns:
        bool: time_convertable_variable feature value
    """
    if time_convertable_variable is None:
        return -1

    return bool(time_convertable_variable)


@dataclass(init=False)
class AnalysisFeaturesSet(FeaturesSet):
    """
    Class to hold analysis features.
    """
    response_variable_label: str = field(
        default="result",
        metadata={
            'gym_space': Discrete(2),
            'get_value_fn': bool
        })

    score: float = field(
        default=0,
        metadata={
            'gym_space': Box(0, 1),
            'get_value_fn': None
        })

    clusters_count: int = field(
        default=0,
        metadata={
            'gym_space': Discrete(100),
            'get_value_fn': None
        })

    time_convertable_variable: str = field(
        default="",
        metadata={
            'gym_space': Discrete(3, start=-1),
            'get_value_fn': time_convertable_variable_to_feature
        })
