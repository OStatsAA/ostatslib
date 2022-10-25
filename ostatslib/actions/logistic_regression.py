"""
Logistic regression module
"""

import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LogisticRegressionCV

from ostatslib.actions.utils import ActionResult, reward_cap, interpretable_model
from ostatslib.states import State


@reward_cap
@interpretable_model
def logistic_regression(state: State, data: DataFrame) -> ActionResult[LogisticRegressionCV]:
    """
    Fits data to a logistic regression model.
    Action result is a ScikitLearn LogisticRegressionCV instance

    Args:
        state (State): current environment state
        data (DataFrame): data under analysis

    Returns:
        tuple[State, float, LogisticRegressionCV]: tuple with next state, reward and \
            ScikitLearn LogisticRegressionCV
    """
    response_var_label = state.get("response_variable_label")
    y_values = data[response_var_label].values
    x_values = data.drop(response_var_label, axis=1).values

    regression = LogisticRegressionCV()

    try:
        regression = regression.fit(x_values, y_values)
    except ValueError:
        return ActionResult(state, -100, None)

    reward = __calculate_reward(state, regression, x_values, y_values)
    return ActionResult(state, reward, regression)


def __calculate_reward(state: State,
                       regression: LogisticRegressionCV,
                       x_values: np.ndarray,
                       y_values: np.ndarray) -> float:
    reward: float = 0

    reward += __penalty_for_continous_response(state)
    reward += __reward_for_accuracy(regression.score(x_values, y_values))

    return reward


def __penalty_for_continous_response(state: State) -> float:
    if state.get("is_response_quantitative") > 0:
        return -100

    return 0


def __reward_for_accuracy(score: float) -> float:
    if score <= .6:
        return - (1 - score) * 100

    if .6 < score <= .9:
        return score * 50

    if score >= .9:
        return score * 60
