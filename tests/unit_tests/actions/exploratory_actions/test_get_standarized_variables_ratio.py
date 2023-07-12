"""
is_standarized_check action testing module
"""

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from ostatslib.actions.exploratory_actions import get_standarized_variables_ratio
from ostatslib.states import State


def test_corr_matrix() -> None:
    """
    Tests get_standarized_variables_ratio
    """
    X, y = load_breast_cancer(return_X_y=True)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    data = DataFrame(X)
    data['target'] = y
    state = State()
    state.set('response_variable_label', 'target')
    state, reward, info = get_standarized_variables_ratio(state, data)
    assert reward > 0
    assert float(state.get('standarized_variables_ratio')) > 0
