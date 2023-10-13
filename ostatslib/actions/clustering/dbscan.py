"""DBSCAN clustering actions module
"""

import operator
import numpy as np
from kneed import KneeLocator
from pandas import DataFrame
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from ostatslib.actions.base import ClusteringEstimatorAction
from ostatslib.config import Config
from ostatslib.states import State


class DBSCANClustering(ClusteringEstimatorAction[DBSCAN]):
    """DBSCAN clustering action.
    Fits a Scikit-Learn DBSCAN.
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
    """

    action_name = 'DBSCAN'
    action_key = 'dbscan'
    estimator = DBSCAN()
    exceptions_handlers = None
    validations = [('clusters_count', operator.eq, 0)]

    def _fit(self, data: DataFrame, state: State, config: Config) -> tuple[DBSCAN, float]:
        max_curvature_point = _get_max_curvature_point(data)
        self.estimator = DBSCAN(eps=max_curvature_point)
        self.estimator = self.estimator.fit(data)
        labels = self.estimator.labels_

        if np.all(labels == -1) or np.all(labels == 0):
            score = 0
        else:
            score: float = silhouette_score(data, labels)

        return self.estimator, score


def _get_max_curvature_point(data: DataFrame) -> float:
    n_rows, n_columns = data.shape
    if n_rows < 2 * n_columns:
        min_points = 5
    elif n_rows < 10 * n_columns:
        min_points = n_columns
    else:
        min_points = 2 * n_columns

    neighbors_fit = NearestNeighbors(
        n_neighbors=min_points, metric='euclidean').fit(data)
    distances, _ = neighbors_fit.kneighbors(data)
    distances = np.sort(distances.sum(axis=1), axis=0)
    elbow_locator = KneeLocator(
        range(0, len(distances)),
        distances,
        curve="convex",
        direction="increasing",
        interp_method='polynomial')

    if elbow_locator.elbow_y is None:
        return 0.5

    return elbow_locator.elbow_y
