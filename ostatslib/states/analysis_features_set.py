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
    __time_convertable_variable: str = ""

    @property
    def time_convertable_variable(self) -> str:
        """
        Feature indicating if there's a variable that may be converted to date format

        Returns:
            str: name of convertable variable
        """
        return self.__time_convertable_variable

    @time_convertable_variable.setter
    def time_convertable_variable(self, value: str | None) -> None:
        self.__time_convertable_variable = value

    def __array__(self):
        return np.array([
            bool(self.response_variable_label),
            self.score,
            bool(self.clusters_count),
            self.__time_convertable_variable_to_feature()
        ])


    def __time_convertable_variable_to_feature(self) -> float:
        if self.__time_convertable_variable is None:
            return -1

        return bool(self.__time_convertable_variable)
