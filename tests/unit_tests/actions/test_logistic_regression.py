# pylint: disable=redefined-outer-name
"""
Logistic regression action testing module
"""

from pandas import DataFrame
from datacooker.recipes import LogitRecipe, Recipe
from datacooker.variables import ContinousVariable
from scipy.stats import norm
import pytest

from ostatslib.actions import logistic_regression
from ostatslib.features_extractors import AnalysisFeaturesSet, DataFeaturesSet
from ostatslib.states import State


@pytest.fixture
def dummy_state() -> State:
    """
    Instantiates a dummy state fixture
    """
    return State(DataFeaturesSet(), AnalysisFeaturesSet(response_variable_label="result"))


@pytest.fixture
def dummy_binary_response_data() -> DataFrame:
    """
    Generates dummy dataset with a binary response variable
    """
    recipe = LogitRecipe(lambda variables, _: 0 + 10 * variables["a"])
    recipe.add_variable(ContinousVariable("a"))
    return recipe.cook()


@pytest.fixture
def dummy_continous_response_data() -> DataFrame:
    """
    Generates dummy dataset with non binary response variable
    """
    recipe = Recipe(lambda variables, error: 0 + 10 * variables["a"] + error)
    recipe.add_variable(ContinousVariable("a"))
    recipe.add_error(lambda _, size: norm().rvs(size=size))
    return recipe.cook()


def test_binary_response_data_yields_positve_reward(dummy_state,
                                                    dummy_binary_response_data) -> None:
    """
    Action should return a positve reward when applied to a linear datatset
    """
    action_result = logistic_regression(
        dummy_state, dummy_binary_response_data)
    assert action_result.reward >= 10


def test_continous_response_data_yields_negative_reward(
        dummy_state: State,
        dummy_continous_response_data: DataFrame) -> None:
    """
    Action should return a positve reward when applied to a linear datatset
    """
    dummy_state.set("is_response_quantitative", 1)
    action_result = logistic_regression(
        dummy_state, dummy_continous_response_data)
    assert action_result.reward <= 0
