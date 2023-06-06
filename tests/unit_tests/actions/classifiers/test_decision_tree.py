# pylint: disable=redefined-outer-name
"""
Decision tree action testing module
"""

from pandas import DataFrame
from datacooker.recipes import LogitRecipe, Recipe
from datacooker.variables import ContinousVariable
from scipy.stats import norm
import pytest

from ostatslib.actions.classifiers import decision_tree
from ostatslib.states import State


@pytest.fixture
def dummy_binary_response_data() -> DataFrame:
    """
    Generates dummy dataset with a binary response variable
    """
    recipe = LogitRecipe(lambda variables, _: 0 + 10 * variables["a"])
    recipe.add_variable(ContinousVariable("a"))
    return recipe.cook()


@pytest.fixture
def dummy_continuous_response_data() -> DataFrame:
    """
    Generates dummy dataset with non binary response variable
    """
    recipe = Recipe(lambda variables, error: 0 + 10 * variables["a"] + error)
    recipe.add_variable(ContinousVariable("a"))
    recipe.add_error(lambda _, size: norm().rvs(size=size))
    return recipe.cook()


def test_binary_response_data_yields_positive_reward(dummy_binary_response_data) -> None:
    """
    Action should return a positive reward when applied to a linear dataset
    """
    state = State()
    state.set("is_response_discrete", 1)
    reward = decision_tree(state, dummy_binary_response_data)[1]
    assert reward >= 0.5


def test_continuous_response_data_yields_negative_reward(
        dummy_continuous_response_data: DataFrame) -> None:
    """
    Action should return a positive reward when applied to a linear dataset
    """
    state = State()
    reward = decision_tree(state, dummy_continuous_response_data)[1]
    assert reward <= -0.5
