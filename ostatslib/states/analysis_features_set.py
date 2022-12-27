"""
AnalysisFeaturesSet  module
"""

from dataclasses import dataclass

import numpy as np


@dataclass(init=False)
class AnalysisFeaturesSet:
    """
    Class to hold analysis features.
    """
    response_variable_label: str = "result"
    score: float = 0
    clusters_count: int = 0

    def __array__(self):
        return np.array([
            bool(self.response_variable_label),
            self.score,
            bool(self.clusters_count)
        ])
