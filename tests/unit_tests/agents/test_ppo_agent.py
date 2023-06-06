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
    agent = PPOAgent(training_envs_count=2)
    agent.train(steps=int(1e2), save_freq=1e2)
    agent.analyze(dummy_dataset)
