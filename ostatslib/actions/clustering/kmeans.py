"""KMeans clustering actions module
"""

import operator
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ostatslib.actions.base import ClusteringEstimatorAction
from ostatslib.config import Config
from ostatslib.states import State


class KMeansClustering(ClusteringEstimatorAction[KMeans]):
    """KMeans clustering action.
    Fits a Scikit-Learn KMeans.
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
    """

    action_name = 'KMeans'
    action_key = 'kmeans'
    estimator = KMeans()
    exceptions_handlers = None
    validations = [('clusters_count', operator.gt, 0)]

    def _fit(self, data: DataFrame, state: State, config: Config) -> tuple[KMeans, float]:
        clusters_count: int = state.get("clusters_count")
        self.estimator = KMeans(n_clusters=clusters_count, n_init='auto')
        self.estimator = self.estimator.fit(data)

        score: float = silhouette_score(data, self.estimator.labels_)

        return self.estimator, score
