"""
Support Vector Classification module
"""

from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from ostatslib.actions import Action, ActionInfo, ActionResult
from ostatslib.actions.utils import (calculate_score_reward,
                                     reward_cap,
                                     opaque_model,
                                     split_response_from_explanatory_variables,
                                     update_state_score)
from ostatslib.states import State

_ACTION_NAME = "Support Vector Classification"


@reward_cap
@opaque_model
def _support_vector_classification(state: State, data: DataFrame) -> ActionResult[SVC]:
    """
    Fits data to a SVC model

    Args:
        state (State): current environment state
        data (DataFrame): data under analysis

    Returns:
        ActionResult[SVC]: action result
    """
    if not __is_valid_state(state):
        return state, -1, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_support_vector_classification,
                                     model=None,
                                     raised_exception=False)

    y_values, x_values = split_response_from_explanatory_variables(state, data)
    classifier = SVC()

    try:
        scores: ndarray = cross_val_score(classifier, x_values, y_values, cv=5)
    except ValueError:
        return state, -1, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_support_vector_classification,
                                     model=None,
                                     raised_exception=True)

    score: float = scores.mean() - scores.std()
    reward: float = calculate_score_reward(score)
    update_state_score(state, score)
    return state, reward, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_support_vector_classification,
                                     model=classifier,
                                     raised_exception=False)


def __is_valid_state(state: State) -> bool:
    if state.get("is_response_quantitative") > 0 or not bool(state.get("response_variable_label")):
        return False

    return True


support_vector_classification: Action[SVC] = _support_vector_classification
