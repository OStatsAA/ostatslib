"""
FeaturesSets classes Module
"""

from dataclasses import dataclass

@dataclass(init=False)
class DataFeaturesSet:
    """
    Class to hold features extracted from a dataset.
    """
    is_response_dichotomous: int = 0
    is_response_quantitative: int = 0
    log_rows_count: float = 0
