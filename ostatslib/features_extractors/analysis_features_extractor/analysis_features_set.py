"""
AnalysisFeaturesSet  module
"""

from dataclasses import dataclass


@dataclass
class AnalysisFeaturesSet:
    """
    Class to hold analysis features.
    """
    response_variable_label: str = "result"
