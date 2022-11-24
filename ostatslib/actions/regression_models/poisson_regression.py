"""
Poisson regression module
"""

import math
from pandas import DataFrame
from statsmodels.api import GLM, families
from statsmodels.genmod.generalized_linear_model import GLMResults

from ostatslib.actions.utils import (ActionResult,
                                     reward_cap,
                                     interpretable_model,
                                     split_response_from_explanatory_variables)
from ostatslib.states import State


@reward_cap
@interpretable_model
def poisson_regression(state: State,
                       data: DataFrame) -> ActionResult[GLMResults]:
    """
    Fits data to a poisson regression model.

    Args:
        state (State): current environment state
        data (DataFrame): data under analysis

    Returns:
        ActionResult[GLMResults]: action result
    """
    if not __is_valid_state(state):
        return ActionResult(state, -1, "PoissonRegression")

    response_var, explanatory_vars = split_response_from_explanatory_variables(state,
                                                                               data)
    try:
        poisson_family = families.Poisson()
        regression: GLMResults = GLM(response_var,
                                     explanatory_vars,
                                     poisson_family).fit()
    except ValueError:
        return ActionResult(state, -1, "PoissonRegression")

    reward = __calculate_reward(regression)
    state = __apply_state_updates(state, regression)
    return ActionResult(state, reward, regression)


def __is_valid_state(state: State) -> bool:
    if state.get("is_response_positive_values_only") <= 0:
        return False

    if state.get("is_response_discrete") <= 0:
        return False

    return True


def __calculate_reward(regression: GLMResults) -> float:
    reward: float = 0
    reward += __reward_for_model_r_squared(regression.pseudo_rsquared())
    return reward


def __reward_for_model_r_squared(rsquared: float) -> float:
    if math.isnan(rsquared):
        return -1

    if rsquared <= .6:
        return - (1 - rsquared)

    return rsquared


def __apply_state_updates(state: State, regression: GLMResults) -> State:
    rsquared = regression.pseudo_rsquared()
    if math.isnan(rsquared):
        return state

    state.set('score', regression.pseudo_rsquared())
    return state
