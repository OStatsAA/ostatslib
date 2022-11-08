# pylint: disable=redefined-outer-name
"""
ReplayMemory testing module
"""
from random import choice, uniform
import pytest

from ostatslib.replay_memories import ReplayMemory
from ostatslib.states import State
from ostatslib.actions import ActionsSpace

ACTIONS_CODES = ActionsSpace().actions_encodings_list


def test_append_and_length() -> None:
    """
    ReplayMemory should be able to append entries.
    Its length is measured by number of entries, not the pre-allocated internal array lengths
    """
    replay_memory = ReplayMemory(2000)
    for _ in range(1000):
        action_code = choice(ACTIONS_CODES)
        reward = uniform(-100, 100)
        replay_memory.append(State().features_vector,
                             action_code,
                             State().features_vector,
                             reward)

    assert len(replay_memory) == 1000


def test_gets_state_action_reward_dict() -> None:
    """
    Returns dictionary with states, actions and rewards (SAR)
    """
    replay_memory = ReplayMemory(2000)
    for _ in range(1000):
        action_code = choice(ACTIONS_CODES)
        reward = uniform(-100, 100)
        replay_memory.append(State().features_vector,
                             action_code,
                             State().features_vector,
                             reward)

    sar_dict = replay_memory.get_sar_entries()
    assert list(sar_dict.keys()) == ["states", "actions", "rewards"]
    assert [len(value) == 1000 for value in sar_dict.values()] == [True] * 3


def test_getting_free_rows_count() -> None:
    """
    Memory should have a method to get free rows count
    """
    replay_memory = ReplayMemory(100)
    for _ in range(90):
        action_code = choice(ACTIONS_CODES)
        reward = uniform(-100, 100)
        replay_memory.append(State().features_vector,
                             action_code,
                             State().features_vector,
                             reward)

    free_rows = replay_memory.get_free_space_count()
    assert free_rows == 10


@pytest.mark.parametrize('max_length, used, expected',
                         [(100, 90, False), (100, 100, True)])
def test_getting_whether_memory_is_full(max_length,
                                        used,
                                        expected) -> None:
    """
    Memory should have a method to get whether it's full or not
    """
    replay_memory = ReplayMemory(max_length)
    for _ in range(used):
        action_code = choice(ACTIONS_CODES)
        reward = uniform(-100, 100)
        replay_memory.append(State().features_vector,
                             action_code,
                             State().features_vector,
                             reward)

    assert replay_memory.is_full() == expected
