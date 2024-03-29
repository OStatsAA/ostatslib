"""
GymEnvironment testing module
"""

import pytest

from ostatslib.actions import ActionsSpace
from ostatslib.environments import GymEnvironment
from ostatslib.environments.data_generators import datacooker_generator


def test_environment_exposes_action_space() -> None:
    """
    Tests if environment is exposing its action space
    """
    env = GymEnvironment()
    assert isinstance(env.action_space, ActionsSpace)


def test_environment_reset() -> None:
    """
    Tests if environment is resettable
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


def test_environment_runs_data_generators_args() -> None:
    """
    Tests if environment runs data generators passed as args
    """
    custom_data_generators_list = [datacooker_generator]
    env = GymEnvironment(data_generators=custom_data_generators_list)
    assert getattr(env, '_data_generators') == custom_data_generators_list


def test_step_when_action_is_none() -> None:
    """
    Tests if environment return same state and minimal reward if action is none
    """
    env = GymEnvironment()
    env.action_space.get_action = lambda x: None  # type: ignore
    state = getattr(env, '_state')
    obs, reward, *_, info = env.step([0])  # type: ignore
    assert obs
    assert reward < 0
    assert state == info.next_state
