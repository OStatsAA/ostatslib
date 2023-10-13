import inspect
import pytest

from ostatslib.actions import (classifiers,
                               regressors,
                               exploratory_actions,
                               clustering,
                               ActionsSpace)
from ostatslib.actions.base import Action


def _is_action(type_: type):
    return inspect.isclass(type_) and issubclass(type_, Action)


@pytest.mark.parametrize('module_',
                         [classifiers, regressors, exploratory_actions, clustering],
                         ids=['classifiers', 'regressors', 'exploratory_actions', 'clustering'])
def test_actions_space_contains_all_actions(module_) -> None:
    actions: list[tuple[str, type[Action]]] = inspect.getmembers(module_, _is_action)
    actions_space = ActionsSpace()
    for _, action in actions:
        actions_space.get_action_by_class(action)
