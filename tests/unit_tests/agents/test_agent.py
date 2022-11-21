# pylint: disable=redefined-outer-name
"""
Agent testing module
"""

from pandas import DataFrame
from datacooker.recipes import LogitRecipe
from datacooker.variables import ContinousVariable
import pytest

from ostatslib.agents import Agent


@pytest.fixture
def dummy_dataset() -> DataFrame:
    """
    Training dataset
    """
    size = 50
    recipe = LogitRecipe(lambda variables, _: 0 + 10 * variables["a"])
    recipe.add_variable(ContinousVariable("a"))
    return recipe.cook(size)


def test_agent_runs_training_episode(dummy_dataset: DataFrame) -> None:
    """
    Tests if agent is able to train on a dataset
    """
    agent = Agent()
    reward = agent.train(dummy_dataset)

    assert isinstance(reward, float)

def test_agent_runs_analysis(dummy_dataset: DataFrame) -> None:
    """
    Tests if agent is able to run an analysis
    """
    agent = Agent()
    agent.train(dummy_dataset)
    analysis = agent.analyze(dummy_dataset)

    assert analysis is not None
