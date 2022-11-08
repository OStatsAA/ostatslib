# pylint: disable=redefined-outer-name
"""
Environment testing module
"""

from random import randrange
from typing import Deque
from pandas import DataFrame
from scipy.stats import norm
from datacooker.recipes import LogitRecipe, Recipe
from datacooker.variables import ContinousVariable
import pytest

from ostatslib.environments import Environment
from ostatslib.features_extractors import AnalysisFeaturesSet, DataFeaturesSet
from ostatslib.states import State


@pytest.fixture
def dummy_state() -> State:
    """
    Instantiates a dummy state fixture
    """
    return State(DataFeaturesSet(), AnalysisFeaturesSet())


@pytest.fixture
def dummy_training_datasets() -> list[DataFrame]:
    """
    Training datasets
    """
    size = 50
    datasets = [None] * size
    for index in range(size):
        recipe = None
        if index % 2:
            recipe = Recipe(lambda variables, error: 0 +
                            10 * variables["a"] + error)
            recipe.add_error(lambda variables, size: norm().rvs(size=size))
        else:
            recipe = LogitRecipe(lambda variables, _: 0 + 10 * variables["a"])
        recipe.add_variable(ContinousVariable("a"))
        datasets[index] = recipe.cook(size=randrange(20, 2000))

    return datasets


def test_environment_train_method(dummy_state: State,
                                  dummy_training_datasets: list[DataFrame]) -> None:
    """
    Tests environment training method
    """
    results = list()
    env = Environment()
    for dataset in dummy_training_datasets:
        results.append(env.train_agent(dummy_state, dataset))
    env.agent.update_model()

    assert len(results) == len(dummy_training_datasets)


def test_environment_run_analysis_method(dummy_state: State,
                                         dummy_training_datasets: list[DataFrame]) -> None:
    """
    Tests environment run analysis method and return a deque of actions
    """
    results = list()
    env = Environment()
    for dataset in dummy_training_datasets:
        results.append(env.train_agent(dummy_state, dataset))
    env.agent.update_model()

    analysis, _ = env.run_analysis(dummy_state, dummy_training_datasets[0])
    assert len(analysis)
