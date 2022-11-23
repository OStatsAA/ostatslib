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
    y_values, x_values = split_response_from_explanatory_variables(state, data)
    regression = LogisticRegressionCV(cv=5)

    try:
        regression = regression.fit(x_values, y_values)
    except ValueError:
        return ActionResult(state, -1, "LogisticRegression")

    score: float = regression.score(x_values, y_values)
    reward = __calculate_reward(state, score)
    state = __apply_state_updates(state, score)
    return ActionResult(state, reward, regression)


def __calculate_reward(state: State, score: float) -> float:
    reward: float = 0

    reward += __penalty_for_continous_response(state)
    reward += __reward_for_accuracy(score)

    return reward


def __penalty_for_continous_response(state: State) -> float:
    is_response_quantitative = state.get("is_response_quantitative") == 1
    is_response_dichotomous = state.get("is_response_dichotomous") == 1

    if is_response_quantitative and not is_response_dichotomous:
        return -1

    return 0


def __reward_for_accuracy(score: float) -> float:
    if score <= .6:
        return - (1 - score)

    return score


def __apply_state_updates(state: State, score: float) -> State:
    state.set('score', score)
    return state
