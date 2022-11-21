# pylint: disable=redefined-outer-name
"""
ActorCritic testing module
"""

from pandas import DataFrame
from datacooker.recipes import LogitRecipe
from datacooker.variables import ContinousVariable
import pytest

from ostatslib.environments import Environment
from ostatslib.reinforcement_learning_methods import ActorCritic
from ostatslib.reinforcement_learning_methods.utils import ModelNotFitError
from ostatslib.states import State


@pytest.fixture
def dummy_dataset() -> DataFrame:
    """
    Training dataset
    """
    size = 50
    recipe = LogitRecipe(lambda variables, _: 0 + 10 * variables["a"])
    recipe.add_variable(ContinousVariable("a"))
    return recipe.cook(size)


def test_actor_critic_run_training(dummy_dataset: DataFrame) -> None:
    """
    Tests if training method runs and updates is_fit flag
    """
    actor_critic = ActorCritic()

    assert not actor_critic.is_fit
    reward = actor_critic.run_training_episode(State(),
                                               dummy_dataset,
                                               Environment(),
                                               max_steps=10)
    assert actor_critic.is_fit
    assert isinstance(reward, float)


def test_actor_critic_run_analysis(dummy_dataset: DataFrame) -> None:
    """
    Tests if method is able to run analysis
    """
    environment = Environment()
    actor_critic = ActorCritic()
    actor_critic.run_training_episode(State(),
                                      dummy_dataset,
                                      environment,
                                      max_steps=10)

    analysis = actor_critic.run_analysis(State(),
                                         dummy_dataset,
                                         environment,
                                         max_steps=10)

    assert analysis is not None


def test_actor_critic_run_analysis_raises_error_if_not_fit(dummy_dataset: DataFrame) -> None:
    """
    Model must've been fitted at least once before running analysis method
    """
    actor_critic = ActorCritic()

    with pytest.raises(ModelNotFitError):
        actor_critic.run_analysis(State(),
                                  dummy_dataset,
                                  Environment(),
                                  max_steps=10)
