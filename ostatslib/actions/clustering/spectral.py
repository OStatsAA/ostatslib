"""Spectral clustering actions module
"""

import operator
from typing import Literal
from pandas import DataFrame
from sklearn.cluster import SpectralClustering as _SpectralClustering
from sklearn.metrics import silhouette_score

from ostatslib.actions.base import ClusteringEstimatorAction
from ostatslib.config import Config
from ostatslib.states import State


class SpectralClustering(ClusteringEstimatorAction[_SpectralClustering]):
    """SpectralClustering clustering action using kmeans to assign labels.
    Fits a Scikit-Learn SpectralClustering.
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering
    """

    action_name = 'Spectral Clustering'
    action_key = 'spectral_clustering'
    estimator = _SpectralClustering()
    exceptions_handlers = None
    validations = [('clusters_count', operator.gt, 0)]
    assign_labels: Literal['kmeans', 'discretize', 'cluster_qr'] = 'kmeans'

    def _fit(self,
             data: DataFrame,
             state: State,
             config: Config) -> tuple[_SpectralClustering, float]:
        clusters_count: int = state.get("clusters_count")
        self.estimator = _SpectralClustering(
            n_clusters=clusters_count, assign_labels=self.assign_labels)
        self.estimator = self.estimator.fit(data)

        score: float = silhouette_score(data, self.estimator.labels_)

        return self.estimator, score


class SpectralDiscretizedLabelingClustering(SpectralClustering):
    """SpectralClustering clustering action discretizing labels assignment.
    Fits a Scikit-Learn SpectralClustering.
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering
    """

    action_name = 'Spectral Clustering with Discretized Labels'
    action_key = 'spectral_clustering_discretize_labels'
    assign_labels = 'discretize'


class SpectralQRLabelingClustering(SpectralClustering):
    """SpectralClustering clustering action cluster_qr labels assignment.
    Fits a Scikit-Learn SpectralClustering.
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering
    """

    action_name = 'Spectral Clustering with Cluster QR Labeling'
    action_key = 'spectral_clustering_qr_labels'
    assign_labels = 'cluster_qr'
