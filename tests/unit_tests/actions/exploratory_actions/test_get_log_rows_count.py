# pylint: disable=redefined-outer-name
"""
get_log_rows_count action testing module
"""

from pandas import DataFrame
from datacooker.recipes import Recipe
from datacooker.variables import ContinousVariable
import pytest

from ostatslib.actions.exploratory_actions import get_log_rows_count
from ostatslib.states import State


@pytest.fixture
def dummy_data() -> DataFrame:
    """
    Generates a dummy dataset
    """
    recipe = Recipe(lambda variables, error: 0 + 10 * variables["a"] + error)
    recipe.add_variable(ContinousVariable("a"))
    return recipe.cook()


def test_log_rows_count_yields_negative_rewards_if_feature_is_known(
        dummy_data: DataFrame) -> None:
    """
    log_rows_count should yield positive rewards if the info is unkown.
    Once the info is known and the state is updated, \
    any other attempt to run the same action should yield a negative reward \
    if the rows count has not changed
    """
    state = State()
    reward = get_log_rows_count(state, dummy_data)[1]
    assert reward > 0

    reward = get_log_rows_count(state, dummy_data)[1]
    assert reward < 0
