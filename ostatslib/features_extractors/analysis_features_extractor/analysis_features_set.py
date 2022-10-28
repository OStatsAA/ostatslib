"""
AnalysisFeaturesSet  module
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class AnalysisFeaturesSet:
    """
    Class to hold analysis features.
    """
    response_variable_label: str = "result"
    score: float = 0

    def __array__(self):
        return np.array([self.score])
