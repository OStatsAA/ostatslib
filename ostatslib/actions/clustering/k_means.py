"""
K-Means module
"""

from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ostatslib.actions import Action, ActionInfo, ActionResult
from ostatslib.actions.utils import (calculate_score_reward,
                                     reward_cap,
                                     update_state_score)
from ostatslib.states import State

_ACTION_NAME = "K-Means"


@reward_cap
def _k_means(state: State, data: DataFrame) -> ActionResult[KMeans]:
    """
    Fits data to a KMeans cluster

    Args:
        state (State): current environment state
        data (DataFrame): data under analysis

    Returns:
        ActionResult[KMeans]: action result
    """
    if not __is_valid_state(state):
        return state, -1, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_k_means,
                                     model=None,
                                     raised_exception=False)

    clusters_count: int = state.get("clusters_count")
    kmeans = KMeans(n_clusters=clusters_count)
    kmeans.fit(data)

    score: float = silhouette_score(data, kmeans.labels_)

    reward: float = calculate_score_reward(score)
    update_state_score(state, score)
    return state, reward, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_k_means,
                                     model=kmeans,
                                     raised_exception=False)


def __is_valid_state(state: State) -> bool:
    if not bool(state.get("clusters_count")) or bool(state.get("response_variable_label")):
        return False

    return True


k_means: Action[KMeans] = _k_means
