# pylint: disable=redefined-outer-name
"""
GymEnvironment testing module
"""

from pandas import DataFrame
from datacooker.recipes import LogitRecipe
from datacooker.variables import ContinousVariable
import pytest

from ostatslib.actions.actions_space import ActionsSpace
from ostatslib.environments import GymEnvironment


@pytest.fixture
def dummy_dataset() -> DataFrame:
    """
    Training datasets
    """
    size = 50
    recipe = LogitRecipe(lambda variables, _: 0 + 10 * variables["a"])
    recipe.add_variable(ContinousVariable("a"))
    return recipe.cook(size)


def test_environment_exposes_action_space() -> None:
    """
    Tests if environment is exposing its action space
    """
    env = GymEnvironment()
    assert isinstance(env.action_space, ActionsSpace)


def test_environment_reset() -> None:
    """
    Tests if environment is resetable
    """
    env = GymEnvironment()
    action = env.action_space.sample()
    step_result = env.step(action)
    assert step_result is not None

def test_environment_implements_render(capsys: pytest.CaptureFixture[str]) -> None:
    """
    Tests if environment is exposing its action space
    """
    env = GymEnvironment()
    env.render()
    captured = capsys.readouterr()
    assert captured.out == "Render has no effect yet\n"
