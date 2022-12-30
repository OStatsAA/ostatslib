"""
FeaturesSets classes Module
"""

from dataclasses import astuple, dataclass

import numpy as np


@dataclass(init=False)
class DataFeaturesSet:
    """
    Class to hold features extracted from a dataset.
    """
    log_rows_count: float = 0
    is_response_dichotomous: int = 0
    is_response_quantitative: int = 0
    is_response_discrete: int = 0
    is_response_positive_values_only: int = 0
    are_linear_model_residuals_correlated: int = 0

    def __array__(self):
        return np.array(astuple(self))
