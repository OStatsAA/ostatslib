# pylint: disable=redefined-outer-name
"""
Environment testing module
"""

from pandas import DataFrame
from datacooker.recipes import LogitRecipe
from datacooker.variables import ContinousVariable
import pytest

from ostatslib.actions.actions_space import ActionsSpace
from ostatslib.actions.utils.action_result import ActionResult
from ostatslib.environments import Environment
from ostatslib.states import State


@pytest.fixture
def dummy_dataset() -> DataFrame:
    """
    Training datasets
    """
    size = 50
    recipe = LogitRecipe(lambda variables, _: 0 + 10 * variables["a"])
    recipe.add_variable(ContinousVariable("a"))
    return recipe.cook(size)


def test_environment_exposes_actions_space() -> None:
    """
    Tests if environment is exposing its actions space
    """
    actions_space = ActionsSpace()
    env = Environment(actions_space=actions_space)
    assert isinstance(env.actions_space, ActionsSpace)


def test_environment_runs_action(dummy_dataset: DataFrame) -> None:
    """
    Tests if environment is able to run a valid action
    """
    actions_space = ActionsSpace()
    env = Environment(actions_space=actions_space)
    valid_action = actions_space.actions_encodings_list[0]
    action_result, done = env.run_action(State(), dummy_dataset, valid_action)

    assert isinstance(action_result, ActionResult)
    assert not done
