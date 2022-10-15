"""
ReplayMemory testing module
"""
from random import choice, uniform
from ostatslib.replay_memories import ReplayMemory
from ostatslib.states.state import State

DUMMY_ACTIONS = ['a', 'b', 'c']


def test_append_and_length() -> None:
    """
    ReplayMemory should be able to append entries.
    Its length is measured by number of entries, not the pre-allocated internal array lengths
    """
    replay_memory = ReplayMemory(2000)
    for _ in range(1000):
        action_name = choice(DUMMY_ACTIONS)
        reward = uniform(-100, 100)
        replay_memory.append(State(), action_name, State(), reward)

    assert len(replay_memory) == 1000


def test_gets_actions_count() -> None:
    """
    ReplayMemory should be able to return each action taken count
    """
    replay_memory = ReplayMemory(2000)
    for i in range(1000):
        action_name = "odd" if i % 2 else choice(DUMMY_ACTIONS)
        reward = uniform(-100, 100)
        replay_memory.append(State(), action_name, State(), reward)

    counter = replay_memory.get_actions_count()
    assert counter.get("odd") == 500


def test_gets_state_action_reward_dict() -> None:
    """
    Returns dictionary with states, actions and rewards (SAR)
    """
    replay_memory = ReplayMemory(2000)
    for _ in range(1000):
        action_name = choice(DUMMY_ACTIONS)
        reward = uniform(-100, 100)
        replay_memory.append(State(), action_name, State(), reward)

    sar_dict = replay_memory.get_sar_entries()
    assert list(sar_dict.keys()) == ["states", "actions", "rewards"]
    assert [len(value) == 1000 for value in sar_dict.values()] == [
        True, True, True]
