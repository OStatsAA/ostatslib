"""
Support Vector Classification module
"""

from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from ostatslib.actions.utils import (ActionResult,
                                     calculate_score_reward,
                                     reward_cap,
                                     opaque_model,
                                     split_response_from_explanatory_variables)
from ostatslib.states import State


@reward_cap
@opaque_model
def support_vector_classification(state: State, data: DataFrame) -> ActionResult[SVC]:
    """
    Fits data to a SVC model

    Args:
        state (State): current environment state
        data (DataFrame): data under analysis

    Returns:
        ActionResult[SVC]: action result
    """
    if not __is_valid_state(state):
        return ActionResult(state, -1, "SVC")

    y_values, x_values = split_response_from_explanatory_variables(state, data)
    classifier = SVC()

    try:
        scores: ndarray = cross_val_score(classifier, x_values, y_values, cv=5)
    except ValueError:
        return ActionResult(state, -1, "SVC")

    score: float = scores.mean() - scores.std()
    reward: float = calculate_score_reward(score)
    state: State = __apply_state_updates(state, score)
    return ActionResult(state, reward, classifier)


def __is_valid_state(state: State) -> bool:
    if state.get("is_response_quantitative") > 0 or not bool(state.get("response_variable_label")):
        return False

    return True


def __apply_state_updates(state: State, score: float) -> State:
    state.set('score', score)
    return state
