"""
AnalysisResult testing module
"""

from ostatslib.actions import ActionInfo
from ostatslib.agents import AnalysisResult
from ostatslib.states import State
from ostatslib.states.analysis_features_set import AnalysisFeaturesSet
from ostatslib.states.data_features_set import DataFeaturesSet


def __action_fn(*args):
    return State(), 0, ActionInfo()


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
    steps = [
        (
            0.5,
            ActionInfo(action_name='Teste',
                       action_fn=__action_fn,
                       model=None,
                       raised_exception=False,
                       state_delta=State(diff_data_features_set))
        ),
        (
            0.9,
            ActionInfo(action_name='Teste',
                       action_fn=__action_fn,
                       model=None,
                       raised_exception=False,
                       state_delta=State(diff_data_features_set))
        ),
    ]
    analysis = AnalysisResult(
        State(analysis_features=initial_analysis_features_set),
        steps,
        True
    )

    assert analysis.summary()
