"""
ModelsFeaturesSet module
"""

from dataclasses import dataclass, field
from gymnasium.spaces import Box

from ostatslib.states.features_set import FeaturesSet


@dataclass(init=False)
class ModelsFeaturesSet(FeaturesSet):
    """
    Class to hold features extracted from a dataset.
    """
    are_linear_model_residuals_correlated: int = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })
    