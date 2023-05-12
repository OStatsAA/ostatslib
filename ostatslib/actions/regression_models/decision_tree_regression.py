"""
Decision Tree Regression module
"""

import operator
from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from ostatslib.states import State
from ..action import Action, ActionInfo, ActionResult
from ..utils import (calculate_score_reward,
                     reward_cap,
                     comprehensible_model,
                     split_response_from_explanatory_variables,
                     update_state_score,
                     validate_state)

_ACTION_NAME = "Decision Tree Regression"
_VALIDATIONS = [('is_response_quantitative', operator.gt, 0),
                ('is_response_dichotomous', operator.lt, 0),
                ('response_variable_label', operator.truth, None)]


@validate_state(action_name=_ACTION_NAME, validator_fns=_VALIDATIONS)
@reward_cap
@comprehensible_model
def _decision_tree_regression(state: State,
                              data: DataFrame) -> ActionResult[DecisionTreeRegressor]:
    """
    Fits data to a decision tree regressor

    Args:
        state (State): current environment state
        data (DataFrame): data under analysis

    Returns:
        ActionResult[DecisionTreeRegressor]: action result
    """
    y_values, x_values = split_response_from_explanatory_variables(state, data)
    classifier = DecisionTreeRegressor()
    scores: ndarray = cross_val_score(classifier, x_values, y_values, cv=5)
    score: float = scores.mean() - scores.std()

    reward: float = calculate_score_reward(score)
    update_state_score(state, score)
    return state, reward, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_decision_tree_regression,
                                     model=classifier,
                                     raised_exception=False)


decision_tree_regression: Action[DecisionTreeRegressor] = _decision_tree_regression
