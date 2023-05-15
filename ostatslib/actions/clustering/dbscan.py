"""
DBSCAN module
"""

from pandas import DataFrame
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from ostatslib import config
from ostatslib.states import State
from ..action import Action, ActionInfo, ActionResult
from ..utils import (calculate_score_reward,
                     reward_cap,
                     update_state_score)

_ACTION_NAME = "DBSCAN"


@reward_cap
def _dbscan(state: State, data: DataFrame) -> ActionResult[DBSCAN]:
    """
    Fits data to a DBSCAN cluster

    Args:
        state (State): current environment state
        data (DataFrame): data under analysis

    Returns:
        ActionResult[DBSCAN]: action result
    """
    if not __is_valid_state(state):
        return state, config.MIN_REWARD, ActionInfo(action_name=_ACTION_NAME,
                                                    action_fn=_dbscan,
                                                    model=None,
                                                    raised_exception=False)

    db_scan = DBSCAN()
    db_scan.fit(data)

    score: float = silhouette_score(data, db_scan.labels_)

    reward: float = calculate_score_reward(score)
    update_state_score(state, score)
    return state, reward, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_dbscan,
                                     model=db_scan,
                                     raised_exception=False)


def __is_valid_state(state: State) -> bool:
    if bool(state.get("clusters_count")) or bool(state.get("response_variable_label")):
        return False

    return True


dbscan: Action[DBSCAN] = _dbscan
