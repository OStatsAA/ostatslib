# pylint: disable=redefined-outer-name
"""
SupportVectorRegression testing module
"""

from unittest.mock import Mock
from sklearn.svm import SVR
from pandas import DataFrame
from datacooker.recipes import LogitRecipe
from datacooker.variables import ContinousVariable
import pytest

from ostatslib.environments import Environment
from ostatslib.reinforcement_learning_methods import SupportVectorRegression
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


@pytest.fixture
def svr_mock() -> Mock:
    """
    SVR mock
    """
    return Mock(wraps=SVR())


def test_svr_run_training(dummy_dataset: DataFrame, svr_mock: Mock) -> None:
    """
    Tests if training method runs and updates is_fit flag
    """
    environment = Environment()
    svr_method = SupportVectorRegression(svr=svr_mock)

    assert not svr_method.is_fit
    reward = svr_method.run_training_episode(State(),
                                             dummy_dataset,
                                             environment,
                                             max_steps=10)
    assert svr_method.is_fit
    assert reward is not None
    svr_mock.fit.assert_called()


def test_svr_run_analysis(dummy_dataset: DataFrame, svr_mock) -> None:
    """
    Tests if method is able to run analysis
    """
    environment = Environment()
    svr_method = SupportVectorRegression(svr=svr_mock)
    svr_method.run_training_episode(State(),
                                    dummy_dataset,
                                    environment,
                                    max_steps=10)

    analysis = svr_method.run_analysis(State(),
                                       dummy_dataset,
                                       environment,
                                       max_steps=10)

    assert analysis is not None


def test_svr_run_analysis_raises_error_if_not_fit(dummy_dataset: DataFrame, svr_mock) -> None:
    """
    Model must've been fitted at least once before running analysis method
    """
    environment = Environment()
    svr_method = SupportVectorRegression(svr=svr_mock)

    with pytest.raises(ModelNotFitError):
        svr_method.run_analysis(State(),
                                dummy_dataset,
                                environment,
                                max_steps=10)
