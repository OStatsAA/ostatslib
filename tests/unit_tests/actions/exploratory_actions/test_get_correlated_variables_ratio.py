"""
get_correlated_variables_ratio action testing module
"""

import numpy as np
from pandas import DataFrame
from ostatslib.actions.exploratory_actions import get_correlated_variables_ratio
from ostatslib.states import State


def test_corr_matrix() -> None:
    """
    Tests get_correlated_variables_ratio
    """
    mean = (1, 1, 2)
    cov = [[1, 0.75, 0], [0.75, 1, 0], [0, 0, 1]]
    expected_ratio = 1/3
    data = np.random.multivariate_normal(mean, cov, 100)
    data = DataFrame(data)
    state = State()
    state, reward, _ = get_correlated_variables_ratio(state, data)
    assert np.isclose(state.get("correlated_variables_ratio"), expected_ratio)
    assert reward > 0

    reward = get_correlated_variables_ratio(state, data)[1]
    assert reward < 0


def test_corr_matrix_when_no_corr() -> None:
    """
    Tests get_correlated_variables_ratio where there's no corr
    """
    mean = (1, 10)
    cov = [[1, 0], [0, 1]]
    expected_state_ratio = -1
    data = np.random.multivariate_normal(mean, cov, 100)
    data = DataFrame(data)
    state = State()
    state, reward, _ = get_correlated_variables_ratio(state, data)
    assert state.get("correlated_variables_ratio") == expected_state_ratio
    assert reward > 0
