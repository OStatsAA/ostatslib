import inspect
from pandas import DataFrame
import pytest
from sklearn.datasets import make_classification

from ostatslib.actions.base import Action
import ostatslib.actions.classifiers.ensembles as ensembles_module
from ostatslib.config import DEFAULT_CONFIG
from ostatslib.states import State


def _is_action_gradient_boosting(type_: type):
    return inspect.isclass(type_) and issubclass(type_, Action) and 'Gradient' in str(type_)


get_grad_boosting_actions = inspect.getmembers(ensembles_module,
                                               _is_action_gradient_boosting)


@pytest.mark.parametrize('action',
                         [action[1]() for action in get_grad_boosting_actions],
                         ids=[str(action[0]) for action in get_grad_boosting_actions])
def test_gradient_boosting_classifiers(action: Action) -> None:
    """Tests GradientBoosting classifiers on a simple data
    """
    X, y = make_classification(n_samples=int(1e3), n_features=5)
    data = DataFrame(X)
    data['target'] = y
    init_state = State()
    init_state.set('response_variable_label', 'target')
    init_state.set('is_response_discrete', 1)
    init_state.set('response_unique_values_ratio', 0.01)
    key = action.action_key + '_score_reward'

    next_state, reward, info = action.execute(
        data, init_state.copy(), DEFAULT_CONFIG)

    assert reward > 0
    assert info.model
    assert init_state.get(key) != next_state.get(key)
