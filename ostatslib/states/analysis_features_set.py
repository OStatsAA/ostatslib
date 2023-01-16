"""
AnalysisFeaturesSet module
"""

from dataclasses import dataclass, field

import numpy as np

from ostatslib.states.features_set import FeaturesSet


@dataclass(init=False)
class AnalysisFeaturesSet(FeaturesSet):
    """
    Class to hold analysis features.
    """
    response_variable_label: str = field(default="result")
    score: float = field(default=0)
    clusters_count: int = field(default=0)
    time_convertable_variable: str = field(default="")

    def __array__(self):
        return np.array([
            bool(self.response_variable_label),
            self.score,
            bool(self.clusters_count),
            self.__time_convertable_variable_to_feature()
        ])

    def __time_convertable_variable_to_feature(self) -> float:
        if self.time_convertable_variable is None:
            return -1

        return bool(self.time_convertable_variable)
