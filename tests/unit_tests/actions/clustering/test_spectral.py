# pylint: disable=redefined-outer-name
"""
KMeans action testing module
"""

import inspect

from pandas import DataFrame
from sklearn.datasets import make_blobs

import pytest

from ostatslib.actions.base import Action
import ostatslib.actions.clustering.spectral as spectral_module
from ostatslib.config import DEFAULT_CONFIG
from ostatslib.states import State

CENTERS = [[1, 1], [-1, -1], [1, -1]]
N_CLUSTERS = len(CENTERS)


def _is_action_spectral_clustering(type_: type):
    return inspect.isclass(type_) and issubclass(type_, Action) and 'Spectral' in str(type_)


get_spectral_clustering_actions = inspect.getmembers(spectral_module,
                                                     _is_action_spectral_clustering)


@pytest.fixture
def dummy_small_deviation_blobs_data() -> DataFrame:
    """
    Generates blobs data
    """
    data, *_ = make_blobs(n_samples=3000,
                          centers=CENTERS,
                          cluster_std=0.1)
    return DataFrame(data)


@pytest.fixture
def dummy_big_deviation_blobs_data() -> DataFrame:
    """
    Generates blobs data
    """
    data, *_ = make_blobs(n_samples=3000,
                          centers=CENTERS,
                          cluster_std=0.8)
    return DataFrame(data)


@pytest.mark.parametrize('action',
                         [action[1]()
                          for action in get_spectral_clustering_actions],
                         ids=[str(action[0]) for action in get_spectral_clustering_actions])
def test_small_deviation_data_yields_positive_reward(dummy_small_deviation_blobs_data: DataFrame,
                                                     action: Action) -> None:
    """
    Action should return a positive reward when applied to a dataset with small deviation
    """
    state = State()
    state.set("response_variable_label", '')
    state.set("clusters_count", N_CLUSTERS)
    key = action.action_key + '_score_reward'

    next_state, reward, info = action.execute(dummy_small_deviation_blobs_data,
                                              state.copy(),
                                              DEFAULT_CONFIG)

    assert reward > 0
    assert info.model
    assert state.get(key) != next_state.get(key)


@pytest.mark.parametrize('action',
                         [action[1]()
                          for action in get_spectral_clustering_actions],
                         ids=[str(action[0]) for action in get_spectral_clustering_actions])
def test_big_deviation_data_yields_negative_reward(dummy_big_deviation_blobs_data: DataFrame,
                                                   action: Action) -> None:
    """
    Action should return a negative reward when applied to a dataset with big deviation
    """
    state = State()
    state.set("response_variable_label", '')
    state.set("clusters_count", N_CLUSTERS)
    key = action.action_key + '_score_reward'

    next_state, reward, info = action.execute(dummy_big_deviation_blobs_data,
                                              state.copy(),
                                              DEFAULT_CONFIG)
    assert reward < 0
    assert info.model
    assert state.get(key) != next_state.get(key)
