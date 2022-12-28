"""
K-Means module
"""

from pandas import DataFrame
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from ostatslib.actions.utils import ActionResult, calculate_score_reward, reward_cap
from ostatslib.states import State


@reward_cap
def dbscan(state: State, data: DataFrame) -> ActionResult[DBSCAN]:
    """
    Fits data to a DBSCAN cluster

    Args:
        state (State): current environment state
        data (DataFrame): data under analysis

    Returns:
        ActionResult[DBSCAN]: action result
    """
    if not __is_valid_state(state):
        return ActionResult(state, -1, "DBSCAN")

    db_scan = DBSCAN()
    db_scan.fit(data)

    score: float = silhouette_score(data, db_scan.labels_)

    reward: float = calculate_score_reward(score)
    state: State = __apply_state_updates(state, score)
    return ActionResult(state, reward, db_scan)


def __is_valid_state(state: State) -> bool:
    if bool(state.get("clusters_count")) or bool(state.get("response_variable_label")):
        return False

    return True


def __apply_state_updates(state: State, score: float) -> State:
    state.set('score', score)
    return state
