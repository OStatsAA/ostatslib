"""
get_missing_data_ratio module
"""

from pandas import DataFrame
from ostatslib import config
from ostatslib.states import State

from ..action import Action, ActionInfo, ActionResult

_ACTION_NAME = "Get Missing Data Ratio"

MISSING_DATA_RATIO_OFFSET = 0.25


def _action(state: State, data: DataFrame) -> ActionResult[None]:
    """
    Gets missing data ratio

    Args:
        state (State): state
        data (DataFrame): data

    Returns:
        ActionResult[None]: action result
    """
    missing_data_count = data.isna().sum().sum()
    missing_ratio = missing_data_count / (data.shape[0] * data.shape[1])
    reward = __calculate_reward(state, missing_ratio)
    __update_state(state, missing_ratio)
    return state, reward, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_action,
                                     model=None,
                                     raised_exception=False)


def __calculate_reward(state: State, missing_data_ratio: float) -> float:
    if state.get("missing_data_ratio") == missing_data_ratio:
        return config.MIN_REWARD

    return config.MAX_EXPLORATORY_REWARD


def __update_state(state: State, missing_data_ratio: float) -> None:
    if not missing_data_ratio:
        return state.set("missing_data_ratio", -1)

    missing_data_ratio += MISSING_DATA_RATIO_OFFSET

    if missing_data_ratio < 1:
        return state.set("missing_data_ratio", missing_data_ratio)

    return state.set("missing_data_ratio", 1)


get_missing_data_ratio: Action[None] = _action
