# pylint: disable=redefined-outer-name
"""
get_response_variable_type action testing module
"""

from pandas import DataFrame
from datacooker.recipes import Recipe
from datacooker.variables import ContinousVariable
import pytest

from ostatslib.actions import get_response_variable_type
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


def test_get_response_var_type_yields_negative_rewards_if_feature_is_known(dummy_state: State,
                                                                           dummy_data) -> None:
    """
    get_response_variable_type should yield positive rewards if the info is unkown.
    Once the info is known and the state is updated, \
    any other attempt to run the same action should yield a negative reward \
    if the features related to response type have not changed
    """
    action_result = get_response_variable_type(dummy_state, dummy_data)
    assert action_result.reward > 0

    action_result = get_response_variable_type(dummy_state, dummy_data)
    assert action_result.reward < 0

    dummy_state.set("is_response_dichotomous", 1)
    action_result = get_response_variable_type(dummy_state, dummy_data)
    assert action_result.reward > 0
