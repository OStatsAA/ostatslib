"""
Logistic regression module
"""

from pandas import DataFrame
from sklearn.linear_model import LogisticRegressionCV
from ostatslib.actions import Action, ActionInfo, ActionResult
from ostatslib.actions.utils import (calculate_score_reward,
                                     reward_cap,
                                     interpretable_model,
                                     split_response_from_explanatory_variables,
                                     update_state_score)
from ostatslib.states import State

_ACTION_NAME = "Logistic Regression"


@reward_cap
@interpretable_model
def _logistic_regression(state: State, data: DataFrame) -> ActionResult[LogisticRegressionCV]:
    """
    Fits data to a logistic regression model.

    Args:
        state (State): current environment state
        data (DataFrame): data under analysis

    Returns:
        ActionResult[LogisticRegressionCV]: action result
    """
    if not __is_valid_state(state):
        return state, -1, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_logistic_regression,
                                     model=None,
                                     raised_exception=False)

    y_values, x_values = split_response_from_explanatory_variables(state, data)
    regression = LogisticRegressionCV(cv=5)

    try:
        regression = regression.fit(x_values, y_values)
    except ValueError:
        return state, -1, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_logistic_regression,
                                     model=None,
                                     raised_exception=True)

    score: float = regression.score(x_values, y_values)
    reward: float = calculate_score_reward(score)
    update_state_score(state, score)
    return state, reward, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_logistic_regression,
                                     model=regression,
                                     raised_exception=False)

def __is_valid_state(state: State) -> bool:
    if state.get("is_response_dichotomous") <= 0:
        return False

    return True


logistic_regression: Action[LogisticRegressionCV] = _logistic_regression
