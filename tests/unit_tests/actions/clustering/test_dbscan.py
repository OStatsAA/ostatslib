# pylint: disable=redefined-outer-name
"""
KMeans action testing module
"""

from pandas import DataFrame
from sklearn.datasets import make_blobs

import pytest

from ostatslib.actions import dbscan
from ostatslib.states import State

CENTERS = [[1, 1], [-1, -1], [1, -1]]
N_CLUSTERS = len(CENTERS)


@pytest.fixture
def dummy_small_deviation_blobs_data() -> DataFrame:
    """
    Generates blobs data
    """
    data, *_ = make_blobs(n_samples=3000,
                          centers=CENTERS,
                          cluster_std=0.1)
    return data


@pytest.fixture
def dummy_big_deviation_blobs_data() -> DataFrame:
    """
    Generates blobs data
    """
    data, *_ = make_blobs(n_samples=3000,
                          centers=CENTERS,
                          cluster_std=0.8)
    return data


def test_small_deviation_data_yields_positive_reward(
        dummy_small_deviation_blobs_data: DataFrame) -> None:
    """
    Action should return a positve reward when applied to a dataset with small deviation
    """
    state = State()
    state.set("response_variable_label", None)
    action_result = dbscan(state, dummy_small_deviation_blobs_data)
    assert action_result.reward >= 0.6


def test_big_deviation_data_yields_negative_reward(
        dummy_big_deviation_blobs_data: DataFrame) -> None:
    """
    Action should return a negative reward when applied to a dataset with big deviation
    """
    state = State()
    state.set("response_variable_label", None)
    action_result = dbscan(state, dummy_big_deviation_blobs_data)
    assert action_result.reward < 0

def test_cluster_count_is_known_yields_negative_reward(
        dummy_small_deviation_blobs_data: DataFrame) -> None:
    """
    Action should return a negative reward if clusters count is known
    """
    state = State()
    state.set("response_variable_label", None)
    state.set("clusters_count", N_CLUSTERS)
    action_result = dbscan(state, dummy_small_deviation_blobs_data)
    assert action_result.reward < 0
