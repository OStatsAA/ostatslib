"""
Logistic regression module
"""

from pandas import DataFrame
from sklearn.linear_model import LogisticRegressionCV

from ostatslib.actions.utils import (ActionResult,
                                     reward_cap,
                                     interpretable_model,
                                     split_response_from_explanatory_variables)
from ostatslib.states import State


@reward_cap
@interpretable_model
def logistic_regression(state: State, data: DataFrame) -> ActionResult[LogisticRegressionCV]:
    """
    Fits data to a logistic regression model.

    Args:
        state (State): current environment state
        data (DataFrame): data under analysis

    Returns:
        ActionResult[LogisticRegressionCV]: action result
    """
    if not __is_valid_state(state):
        return ActionResult(state, -1, "LogisticRegression")

    y_values, x_values = split_response_from_explanatory_variables(state, data)
    regression = LogisticRegressionCV(cv=5)

    try:
        regression = regression.fit(x_values, y_values)
    except ValueError:
        return ActionResult(state, -1, "LogisticRegression")

    score: float = regression.score(x_values, y_values)
    reward = __calculate_reward(score)
    state = __apply_state_updates(state, score)
    return ActionResult(state, reward, regression)


def __is_valid_state(state: State) -> bool:
    if state.get("is_response_dichotomous") <= 0:
        return False

    return True


def __calculate_reward(score: float) -> float:
    reward: float = 0
    reward += __reward_for_accuracy(score)
    return reward


def __reward_for_accuracy(score: float) -> float:
    if score <= .6:
        return - (1 - score)

    return score


def __apply_state_updates(state: State, score: float) -> State:
    state.set('score', score)
    return state
