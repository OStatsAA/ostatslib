"""
Decision Tree module
"""

from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from ostatslib.actions.utils import (ActionResult,
                                     reward_cap,
                                     comprehensible_model,
                                     split_response_from_explanatory_variables)
from ostatslib.states import State


@reward_cap
@comprehensible_model
def decision_tree(state: State,
                  data: DataFrame) -> ActionResult[DecisionTreeClassifier]:
    """
    Fits data to a decision tree classifier

    Args:
        state (State): current environment state
        data (DataFrame): data under analysis

    Returns:
        ActionResult[DecisionTreeClassifier]: action result
    """
    if not __is_valid_state(state):
        return ActionResult(state, -1, "DecisionTreeClassifier")

    y_values, x_values = split_response_from_explanatory_variables(state, data)
    classifier = DecisionTreeClassifier()
    scores: ndarray = cross_val_score(classifier, x_values, y_values, cv=5)
    score: float = scores.mean() - scores.std()

    reward = __calculate_reward(score)
    state = __apply_state_updates(state, score)
    return ActionResult(state, reward, classifier.fit(X=x_values, y=y_values))


def __is_valid_state(state: State) -> bool:
    if state.get("is_response_quantitative") > 0 or not bool(state.get("response_variable_label")):
        return False

    return True


def __calculate_reward(score: float) -> float:
    if score <= .6:
        return - (1 - score)

    return score


def __apply_state_updates(state: State, score: float) -> State:
    state.set('score', score)
    return state
