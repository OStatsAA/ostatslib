# pylint: disable=redefined-outer-name
"""
Linear regression action testing module
"""

from pandas import DataFrame
from datacooker.recipes import LogitRecipe, Recipe
from datacooker.variables import ContinousVariable
from scipy.stats import norm
from statsmodels.regression.linear_model import RegressionResults
import pytest

from ostatslib.actions import linear_regression
from ostatslib.actions.utils import ActionResult
from ostatslib.features_extractors import AnalysisFeaturesSet, DataFeaturesSet
from ostatslib.states import State


@pytest.fixture
def dummy_state() -> State:
    """
    Instantiates a dummy state fixture
    """
    return State(DataFeaturesSet(), AnalysisFeaturesSet(response_variable_label="result"))


@pytest.fixture
def dummy_linear_data() -> DataFrame:
    """
    Generates a linear dummy dataset
    """
    recipe = Recipe(lambda variables, error: 0 + 10 * variables["a"] + error)
    recipe.add_variable(ContinousVariable("a"))
    recipe.add_error(lambda variables, size: norm().rvs(size=size))
    return recipe.cook()


@pytest.fixture
def dummy_binary_response_data() -> DataFrame:
    """
    Generates dummy dataset with a binary response variable
    """
    recipe = LogitRecipe(lambda variables, _: 0 + 10 * variables["a"])
    recipe.add_variable(ContinousVariable("a"))
    return recipe.cook()


def test_linear_data_yields_positve_reward(dummy_state, dummy_linear_data) -> None:
    """
    Action should return a positve reward when applied to a linear datatset
    """
    action_result: ActionResult[RegressionResults] = linear_regression(dummy_state,
                                                                       dummy_linear_data)
    assert action_result.reward >= 10


def test_binary_data_yields_negative_reward(dummy_state, dummy_binary_response_data) -> None:
    """
    Action should return a positve reward when applied to a linear datatset
    """
    dummy_state.set("is_response_dichotomous", 1)
    action_result: ActionResult[RegressionResults] = linear_regression(dummy_state,
                                                                       dummy_binary_response_data)
    assert action_result.reward <= 0
