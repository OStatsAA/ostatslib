"""
Support Vector Classification module
"""

import operator
from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from ostatslib import config
from ostatslib.states import State

from ..action import Action, ActionInfo, ActionResult
from ..utils import (calculate_score_reward,
                     reward_cap,
                     opaque_model,
                     split_response_from_explanatory_variables,
                     update_state_score,
                     validate_state)

_ACTION_NAME = "Support Vector Classification"
_VALIDATIONS = [('is_response_quantitative', operator.lt, 0),
                ('response_variable_label', operator.truth, None)]


@validate_state(action_name=_ACTION_NAME, validator_fns=_VALIDATIONS)
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
    y_values, x_values = split_response_from_explanatory_variables(state, data)
    classifier = SVC()

    try:
        scores: ndarray = cross_val_score(classifier, x_values, y_values, cv=5)
    except ValueError:
        return state, config.MIN_REWARD, ActionInfo(action_name=_ACTION_NAME,
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


support_vector_classification: Action[SVC] = _support_vector_classification
