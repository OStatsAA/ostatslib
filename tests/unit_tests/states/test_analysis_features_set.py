"""
AnalysisFeaturesSet testing module
"""

from ostatslib.states.analysis_features_set import AnalysisFeaturesSet


def test_time_convertable_when_None() -> None:
    """
    Tests if time_convertible_variable_to_feature returns -1 if None is set
    """
    features_set = AnalysisFeaturesSet()
    features_set.time_convertible_variable = None

    assert features_set.as_features_dict()['time_convertible_variable'] == [-1]
