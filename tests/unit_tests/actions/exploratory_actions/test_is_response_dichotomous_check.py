# pylint: disable=redefined-outer-name
"""
is_response_dichotomous_check action testing module
"""

from pandas import DataFrame
from datacooker.recipes import LogitRecipe
from datacooker.variables import ContinousVariable
import numpy as np
import pytest

from ostatslib.actions import is_response_dichotomous_check
from ostatslib.states import State


@pytest.fixture
def dummy_data() -> DataFrame:
    """
    Generates a dummy dataset
    """
    recipe = LogitRecipe(lambda variables, error: 0 + variables["a"])
    recipe.add_variable(ContinousVariable("a"))
    return recipe.cook()


def test_is_response_dichotomous_check_yields_negative_rewards_if_feature_is_known(
        dummy_data: DataFrame) -> None:
    """
    is_response_dichotomous_check should yield positive rewards if the info is unkown.
    Once the info is known and the state is updated, \
    any other attempt to run the same action should yield a negative reward
    """
    state = State()
    action_result = is_response_dichotomous_check(state, dummy_data)
    assert action_result.reward > 0
    assert state.get("is_response_dichotomous") == 1

    action_result = is_response_dichotomous_check(state, dummy_data)
    assert action_result.reward < 0


@pytest.mark.parametrize('result_type, result, expected',
                         [('mixed-integer', ['a', 1], 1),
                          ('mixed-bool-integer', [False, 1], 1),
                          ('mixed-integer-float', [1, 2.5], -1)]
                         )
def test_mixed_dichotomous_values(result_type: str, result: list, expected: bool) -> None:
    """
    Mixed dichotomous dtypes
    """
    state = State()
    data = DataFrame({
        'x1': np.random.standard_normal(size=100),
        'result': result * 50
    })
    is_response_dichotomous_check(state, data)
    assert state.get("is_response_dichotomous") == expected
