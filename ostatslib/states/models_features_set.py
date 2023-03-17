"""
ModelsFeaturesSet module
"""

from dataclasses import dataclass, field
from gymnasium.spaces import Box

from ostatslib.states.features_set import FeaturesSet


@dataclass(init=False)
class ModelsFeaturesSet(FeaturesSet):
    """
    Class to hold features extracted from models fitting attempts.
    """
    are_linear_model_regression_residuals_correlated: int = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    are_linear_model_regression_residuals_heteroscedastic: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    are_linear_model_regression_residuals_normally_distributed: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })
    
    is_linear_model_regression_recursive_residuals_mean_zero: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    does_poisson_regression_raises_perfect_separation_error: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })
    