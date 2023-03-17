from pandas import DataFrame
from datacooker.recipes import LogitRecipe
from datacooker.variables import ContinousVariable
import pytest

from ostatslib.agents.ppo_agent import PPOAgent


@pytest.fixture
def dummy_dataset() -> DataFrame:
    """
    Training dataset
    """
    size = 50
    recipe = LogitRecipe(lambda variables, _: 0 + 10 * variables["a"])
    recipe.add_variable(ContinousVariable("a"))
    return recipe.cook(size)

def test_training(dummy_dataset: DataFrame) -> None:
    agent = PPOAgent()
    agent.train(steps=1e3)
    agent.analyze(dummy_dataset)
