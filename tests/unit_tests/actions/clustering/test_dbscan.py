# pylint: disable=redefined-outer-name
"""
DBSCAN action testing module
"""

from pandas import DataFrame
from sklearn.datasets import make_blobs

import pytest

from ostatslib.actions.clustering import DBSCANClustering
from ostatslib.config import DEFAULT_CONFIG
from ostatslib.states import State

CENTERS = [[1, 1], [-1, -1], [1, -1]]
N_CLUSTERS = len(CENTERS)


@pytest.fixture
def dummy_small_deviation_blobs_data() -> DataFrame:
    """
    Generates blobs data
    """
    data, *_ = make_blobs(n_samples=300,
                          centers=CENTERS,
                          cluster_std=0.1)
    return DataFrame(data)


@pytest.fixture
def dummy_big_deviation_blobs_data() -> DataFrame:
    """
    Generates blobs data
    """
    data, *_ = make_blobs(n_samples=300,
                          centers=CENTERS,
                          cluster_std=0.9)
    return DataFrame(data)


def test_small_deviation_data_yields_positive_reward(
        dummy_small_deviation_blobs_data: DataFrame) -> None:
    """
    Action should return a positive reward when applied to a dataset with small deviation
    """
    state = State()
    state.set("response_variable_label", '')
    reward = DBSCANClustering().execute(dummy_small_deviation_blobs_data,
                                        state,
                                        DEFAULT_CONFIG)[1]
    assert reward >= 0.6


def test_big_deviation_data_yields_negative_reward(
        dummy_big_deviation_blobs_data: DataFrame) -> None:
    """
    Action should return a negative reward when applied to a dataset with big deviation
    """
    state = State()
    state.set("response_variable_label", '')
    reward = DBSCANClustering().execute(dummy_big_deviation_blobs_data,
                                        state,
                                        DEFAULT_CONFIG)[1]
    assert reward < 0


def test_cluster_count_is_known_yields_negative_reward(
        dummy_small_deviation_blobs_data: DataFrame) -> None:
    """
    Action should return a negative reward if clusters count is known
    """
    state = State()
    state.set("response_variable_label", '')
    state.set("clusters_count", N_CLUSTERS)
    reward = DBSCANClustering().execute(dummy_small_deviation_blobs_data,
                                        state,
                                        DEFAULT_CONFIG)[1]
    assert reward < 0
