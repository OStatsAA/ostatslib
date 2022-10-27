# pylint: disable=redefined-outer-name
"""
get_log_rows_count action testing module
"""

from pandas import DataFrame
from datacooker.recipes import Recipe
from datacooker.variables import ContinousVariable
import pytest

from ostatslib.actions import get_log_rows_count
from ostatslib.features_extractors import AnalysisFeaturesSet, DataFeaturesSet
from ostatslib.states import State


@pytest.fixture
def dummy_state() -> State:
    """
    Instantiates a dummy state fixture
    """
    return State(DataFeaturesSet(), AnalysisFeaturesSet())


@pytest.fixture
def dummy_data() -> DataFrame:
    """
    Generates a dummy dataset
    """
    recipe = Recipe(lambda variables, error: 0 + 10 * variables["a"] + error)
    recipe.add_variable(ContinousVariable("a"))
    return recipe.cook()


def test_log_rows_count_yields_negative_rewards_if_feature_is_known(dummy_state,
                                                                    dummy_data) -> None:
    """
    log_rows_count should yield positive rewards if the info is unkown.
    Once the info is known and the state is updated, \
    any other attempt to run the same action should yield a negative reward \
    if the rows count hasn't changed
    """
    action_result = get_log_rows_count(dummy_state, dummy_data)
    assert action_result.reward > 0

    action_result = get_log_rows_count(dummy_state, dummy_data)
    assert action_result.reward < 0
