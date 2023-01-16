"""
DataFeaturesSets module
"""

from dataclasses import astuple, dataclass, field

import numpy as np

from ostatslib.states.features_set import FeaturesSet


@dataclass(init=False)
class DataFeaturesSet(FeaturesSet):
    """
    Class to hold features extracted from a dataset.
    """
    log_rows_count: float = field(default=0)
    is_response_dichotomous: int = field(default=0)
    is_response_quantitative: int = field(default=0)
    is_response_discrete: int = field(default=0)
    is_response_positive_values_only: int = field(default=0)
    are_linear_model_residuals_correlated: int = field(default=0)

    def __array__(self):
        return np.array(astuple(self))
