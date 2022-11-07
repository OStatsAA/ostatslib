# pylint: disable=redefined-outer-name
"""
SupportVectorRegression testing module
"""

from random import choice, random
from unittest.mock import Mock
from sklearn.svm import SVR
import numpy as np
import pytest


from ostatslib.actions import ActionsSpace
from ostatslib.features_extractors import AnalysisFeaturesSet, DataFeaturesSet
from ostatslib.reinforcement_learning_models import SupportVectorRegression
from ostatslib.states import State


@pytest.fixture
def dummy_state() -> State:
    """
    Instantiates a dummy state fixture
    """
    return State(DataFeaturesSet(), AnalysisFeaturesSet())


@pytest.fixture
def dummy_training_dataset() -> State:
    """
    Instantiates a dummy training dataset
    """
    size = 10
    state = State(DataFeaturesSet(), AnalysisFeaturesSet())
    actions_space = ActionsSpace()
    actions_list = actions_space.actions_encodings_list
    encoding_length = actions_space.encoding_length

    states_features = [None] * size
    actions_features = np.ndarray(shape=(size, encoding_length))
    rewards = np.ndarray(shape=size)

    for index in range(size):
        state.set("log_rows_count", random())
        states_features[index] = state.features_vector
        actions_features[index] = choice(actions_list)
        rewards[index] = random()

    return np.asarray(states_features), actions_features, rewards


@pytest.fixture
def svr_mock() -> Mock:
    """
    SVR mock
    """
    return Mock(wraps=SVR())


def test_svr_fitting(dummy_training_dataset: tuple, svr_mock: Mock) -> None:
    """
    Tests if fitting method updates is_fit flag and calls underlying model.
    """
    svr_model = SupportVectorRegression(svr=svr_mock)
    assert not svr_model.is_fit

    svr_model.fit(*dummy_training_dataset)

    assert svr_model.is_fit
    svr_mock.fit.assert_called_once()


def test_svr_predicting(dummy_training_dataset: tuple, dummy_state: State) -> None:
    """
    Tests if predicting method return a valid callable action in actions space.
    """
    svr_model = SupportVectorRegression()
    svr_model.fit(*dummy_training_dataset)
    dummy_state.set("log_rows_count", .5)
    predicted = svr_model.predict(dummy_training_dataset[0],
                                  dummy_training_dataset[1])

    assert isinstance(predicted, np.ndarray)
    assert callable(ActionsSpace().get_action_by_encoding(predicted))
