# pylint: disable=redefined-outer-name
"""
Agent testing module
"""

from unittest.mock import Mock
from numpy import array_equal, ndarray
import pytest

from ostatslib.agents import Agent
from ostatslib.exploration_strategies import EpsilonGreedy
from ostatslib.features_extractors import AnalysisFeaturesSet, DataFeaturesSet
from ostatslib.replay_memories import ReplayMemory
from ostatslib.states import State

TEST_ACTION_CODE = ndarray([0])


@pytest.fixture
def dummy_state() -> State:
    """
    Instantiates a dummy state fixture
    """
    return State(DataFeaturesSet(), AnalysisFeaturesSet())


@pytest.fixture
def model_mock() -> Mock:
    """
    Reinforcement Learning model mock
    """
    mock = Mock()
    attrs = {
        'fit.return_value': None,
        'predict.return_value': TEST_ACTION_CODE
    }
    mock.configure_mock(**attrs)
    return mock


@pytest.fixture
def exploration_strategy_mock() -> Mock:
    """
    Exploration strategy mock
    """
    return Mock(wraps=EpsilonGreedy())


def test_agent_memory(dummy_state: State) -> None:
    """
    Tests if agent is saving transitions to memory
    """
    agent = Agent()
    assert agent.is_memory_full is False
    assert agent.memory_length == 0

    agent.remember_transition(dummy_state, TEST_ACTION_CODE, dummy_state, .42)

    assert agent.is_memory_full is False
    assert agent.memory_length == 1


def test_agent_calls_model_fit_on_update(model_mock: Mock) -> None:
    """
    Agent update should call its internal model fit method passsing agent's memory as argument
    """
    memory = ReplayMemory()
    agent = Agent(model=model_mock, replay_memory=memory)
    agent.update_model()

    model_mock.fit.assert_called_once


def test_agent_get_action_method(dummy_state: State,
                                 model_mock: Mock,
                                 exploration_strategy_mock: Mock) -> None:
    """
    Agent get_action method should call its internal model and return model's prediction
    """
    agent = Agent(model=model_mock,
                  exploration_strategy=exploration_strategy_mock)
    action_code = agent.get_action(dummy_state, [TEST_ACTION_CODE])

    assert array_equal(action_code, TEST_ACTION_CODE)
    model_mock.predict.assert_called_once()
    exploration_strategy_mock.get_action.assert_not_called()


def test_agent_get_action_method_while_training(dummy_state: State,
                                                model_mock: Mock,
                                                exploration_strategy_mock: Mock) -> None:
    """
    Agent get_action method should call its exploration strategy get_action method while training
    """
    agent = Agent(model=model_mock,
                  is_training=True,
                  exploration_strategy=exploration_strategy_mock)
    action_code = agent.get_action(dummy_state, [TEST_ACTION_CODE])

    assert array_equal(action_code, TEST_ACTION_CODE)
    exploration_strategy_mock.get_action.assert_called_once()
