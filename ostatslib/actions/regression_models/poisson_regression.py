"""
Poisson regression module
"""

from pandas import DataFrame
from statsmodels.api import GLM, families
from statsmodels.genmod.generalized_linear_model import GLMResults
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from ostatslib.actions import Action, ActionInfo, ActionResult
from ostatslib.actions.utils import (calculate_score_reward,
                                     reward_cap,
                                     interpretable_model,
                                     split_response_from_explanatory_variables,
                                     update_state_score)
from ostatslib.states import State

_ACTION_NAME = "Poisson Regression"


@reward_cap
@interpretable_model
def _poisson_regression(state: State, data: DataFrame) -> ActionResult[GLMResults]:
    """
    Fits data to a poisson regression model.

    Args:
        state (State): current environment state
        data (DataFrame): data under analysis

    Returns:
        ActionResult[GLMResults]: action result
    """
    if not __is_valid_state(state):
        return state, -1, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_poisson_regression,
                                     model=None,
                                     raised_exception=False)

    response_var, explanatory_vars = split_response_from_explanatory_variables(state,
                                                                               data)
    try:
        poisson_family = families.Poisson()
        regression: GLMResults = GLM(response_var,
                                     explanatory_vars,
                                     poisson_family).fit()
    except ValueError:
        return state, -1, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_poisson_regression,
                                     model=None,
                                     raised_exception=True)
    except PerfectSeparationError:
        state.set('does_poisson_regression_raises_perfect_separation_error', 1)
        state.set('score', -1)
        return state, -1, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_poisson_regression,
                                     model=None,
                                     raised_exception=True)

    state.set('does_poisson_regression_raises_perfect_separation_error', -1)
    reward = __calculate_reward(regression)
    state = update_state_score(state, regression.pseudo_rsquared())
    return state, reward, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_poisson_regression,
                                     model=regression,
                                     raised_exception=False)


def __is_valid_state(state: State) -> bool:
    if state.get("is_response_positive_values_only") <= 0:
        return False

    if state.get("is_response_discrete") <= 0:
        return False

    return True


def __calculate_reward(regression: GLMResults) -> float:
    reward: float = 0
    reward += calculate_score_reward(regression.pseudo_rsquared())
    return reward


poisson_regression: Action[GLMResults] = _poisson_regression
