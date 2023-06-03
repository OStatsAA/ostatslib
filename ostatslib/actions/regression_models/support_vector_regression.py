"""
Support Vector Regression module
"""

import operator
from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from ostatslib import config

from ostatslib.states import State
from ..action import Action, ActionInfo, ActionResult
from ..utils import (calculate_score_reward,
                     reward_cap,
                     opaque_model,
                     split_response_from_explanatory_variables,
                     update_state_score,
                     validate_state)

_ACTION_NAME = "Support Vector Regression"
_VALIDATIONS = [('is_response_quantitative', operator.gt, 0),
                ('is_response_dichotomous', operator.gt, 0),
                ('response_variable_label', operator.truth, None),
                ('log_rows_count', operator.gt, 0),
                ('log_rows_count', operator.lt, 0.81)]


@validate_state(action_name=_ACTION_NAME, validator_fns=_VALIDATIONS)
@reward_cap
@opaque_model
def _support_vector_regression(state: State, data: DataFrame) -> ActionResult[SVR]:
    """
    Fits data to a SVR model

    Args:
        state (State): current environment state
        data (DataFrame): data under analysis

    Returns:
        ActionResult[SVR]: action result
    """
    y_values, x_values = split_response_from_explanatory_variables(state, data)
    classifier = SVR()

    try:
        scores: ndarray = cross_val_score(classifier, x_values, y_values, cv=5)
    except ValueError:
        return state, config.MIN_REWARD, ActionInfo(action_name=_ACTION_NAME,
                                                    action_fn=_support_vector_regression,
                                                    model=None,
                                                    raised_exception=True)

    score: float = scores.mean() - scores.std()
    update_state_score(state, score)
    reward = calculate_score_reward(score)
    state.set('support_vector_regression_score_reward', reward)
    return state, reward, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_support_vector_regression,
                                     model=classifier,
                                     raised_exception=False)


support_vector_regression: Action[SVR] = _support_vector_regression
