"""
AnalysisResult testing module
"""

from ostatslib.actions import ActionInfo
from ostatslib.agents import AnalysisResult
from ostatslib.states import State
from ostatslib.states.analysis_features_set import AnalysisFeaturesSet
from ostatslib.states.data_features_set import DataFeaturesSet


def test_analysis_summary() -> None:
    """
    Tests if analysis summary runs
    """
    initial_analysis_features_set = AnalysisFeaturesSet()
    initial_analysis_features_set.response_variable_label = "test"
    diff_data_features_set = DataFeaturesSet()
    diff_data_features_set.is_response_dichotomous = 1
    diff_data_features_set.is_response_positive_values_only = 1
    diff_analysis_features_set = AnalysisFeaturesSet()
    diff_analysis_features_set.score = 0.9
    steps = [(0.5, ActionInfo(action_name='Test', next_state=State())),
             (0.9, ActionInfo(action_name='Test', next_state=State()))]
    analysis = AnalysisResult(State(analysis_features=initial_analysis_features_set),
                              steps,
                              True)

    assert analysis.summary()
    assert analysis.steps_count == len(steps)
    assert analysis.actions_names_list == ['Test', 'Test']
