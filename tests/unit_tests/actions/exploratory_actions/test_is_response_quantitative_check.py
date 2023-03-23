# pylint: disable=redefined-outer-name
"""
is_response_quantitative_check action testing module
"""

from pandas import DataFrame
from datacooker.recipes import Recipe
from datacooker.variables import ContinousVariable
import pytest

from ostatslib.actions import is_response_quantitative_check
from ostatslib.states import State


@pytest.fixture
def dummy_data() -> DataFrame:
    """
    Generates a dummy dataset
    """
    recipe = Recipe(lambda variables, error: 0 + variables["a"])
    recipe.add_variable(ContinousVariable("a"))
    return recipe.cook()


def test_is_response_quantitative_check_yields_negative_rewards_if_feature_is_known(
        dummy_data: DataFrame) -> None:
    """
    is_response_quantitative_check should yield positive rewards if the info is unkown.
    Once the info is known and the state is updated, \
    any other attempt to run the same action should yield a negative reward
    """
    state = State()
    state, reward, _ = is_response_quantitative_check(state, dummy_data)
    assert reward > 0
    assert state.get("is_response_quantitative") == 1

    reward = is_response_quantitative_check(state, dummy_data)[1]
    assert reward < 0
