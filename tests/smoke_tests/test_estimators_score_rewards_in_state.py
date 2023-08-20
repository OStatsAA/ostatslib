import inspect
import pytest

from ostatslib.actions import classifiers, regressors
from ostatslib.actions.base import Action
from ostatslib.states import State


def _is_action(type_: type):
    return inspect.isclass(type_) and issubclass(type_, Action)


@pytest.mark.parametrize('module_',
                         [classifiers, regressors],
                         ids=['classifiers', 'regressors'])
def test_estimators_score_rewards_in_state(module_) -> None:
    actions: list[tuple[str, type[Action]]
                  ] = inspect.getmembers(module_, _is_action)
    state = State()
    for _, action in actions:
        assert state.get(action.action_key + "_score_reward") == 0


def test_no_unmatched_score_rewards_in_state() -> None:
    modules = [classifiers, regressors]
    actions: list[tuple[str, type[Action]]] = []
    for module_ in modules:
        actions.extend(inspect.getmembers(module_, _is_action))

    state_score_reward_keys = [
        key for key in State().keys if key.endswith('_score_reward')
    ]

    unmatched_keys = []
    for key in state_score_reward_keys:
        found_match = False
        for _, action in actions:
            if key == action.action_key + "_score_reward":
                found_match = True
                break

        if not found_match:
            unmatched_keys.append(key)

    assert not unmatched_keys
