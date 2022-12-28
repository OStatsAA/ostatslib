"""
K-Means module
"""

from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ostatslib.actions.utils import ActionResult, calculate_score_reward, reward_cap
from ostatslib.states import State


@reward_cap
def k_means(state: State, data: DataFrame) -> ActionResult[KMeans]:
    """
    Fits data to a KMeans cluster

    Args:
        state (State): current environment state
        data (DataFrame): data under analysis

    Returns:
        ActionResult[KMeans]: action result
    """
    if not __is_valid_state(state):
        return ActionResult(state, -1, "KMeans")

    clusters_count: int = state.get("clusters_count")
    kmeans = KMeans(n_clusters=clusters_count)
    kmeans.fit(data)

    score: float = silhouette_score(data, kmeans.labels_)

    reward: float = calculate_score_reward(score)
    state: State = __apply_state_updates(state, score)
    return ActionResult(state, reward, kmeans)


def __is_valid_state(state: State) -> bool:
    if not bool(state.get("clusters_count")) or bool(state.get("response_variable_label")):
        return False

    return True


def __apply_state_updates(state: State, score: float) -> State:
    state.set('score', score)
    return state
