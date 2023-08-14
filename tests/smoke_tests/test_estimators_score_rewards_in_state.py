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
    actions: list[tuple[str, Action]] = inspect.getmembers(module_, _is_action)
    state = State()
    for _, action in actions:
        assert state.get(action.action_key + "_score_reward") == 0
