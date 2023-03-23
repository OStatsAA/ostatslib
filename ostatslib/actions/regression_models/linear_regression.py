"""
Linear regression module

ref:
https://www.kirenz.com/post/2021-11-14-linear-regression-diagnostics-in-python/linear-regression-diagnostics-in-python/
"""

from math import nan
import numpy as np
from pandas import DataFrame
from statsmodels.api import OLS
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan, linear_harvey_collier
from statsmodels.regression.linear_model import RegressionResults
from ostatslib.actions import Action, ActionInfo, ActionResult
from ostatslib.actions.utils import (calculate_score_reward,
                                     reward_cap,
                                     interpretable_model,
                                     split_response_from_explanatory_variables,
                                     update_state_score)
from ostatslib.states import State

_ACTION_NAME = "Linear Regression"


@reward_cap
@interpretable_model
def _linear_regression(state: State, data: DataFrame) -> ActionResult[RegressionResults]:
    """
    Fits data to a linear regression model.

    Args:
        state (State): current environment state
        data (DataFrame): data under analysis

    Returns:
        ActionResult[RegressionResults]: action result
    """
    if not __is_valid_state(state):
        return state, -1, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_linear_regression,
                                     model=None,
                                     raised_exception=False)

    response_var, explanatory_vars = split_response_from_explanatory_variables(
        state, data)
    try:
        regression: RegressionResults = OLS(response_var, explanatory_vars).fit()
    except ValueError:
        return state, -1, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_linear_regression,
                                     model=None,
                                     raised_exception=True)

    reward = __calculate_reward(state, regression)
    update_state_score(state, regression.rsquared)
    return state, reward, ActionInfo(action_name=_ACTION_NAME,
                                     action_fn=_linear_regression,
                                     model=regression,
                                     raised_exception=False)


def __is_valid_state(state: State) -> bool:
    if state.get("is_response_quantitative") <= 0:
        return False

    return True


def __calculate_reward(state: State, regression: RegressionResults) -> float:
    explanatory_vars: np.ndarray = regression.model.exog
    residuals: np.ndarray = regression.resid.values
    reward: float = 0

    reward += __reward_for_normally_distributed_errors(state, regression)
    reward += __penalty_for_correlation_of_error_terms(state, residuals)
    reward += __reward_for_homoscedasticity(state, residuals, explanatory_vars)
    reward += __reward_for_recursive_residuals_mean(state, regression)
    reward += calculate_score_reward(regression.rsquared)

    return reward


def __reward_for_normally_distributed_errors(state: State,
                                             regression: RegressionResults) -> float:
    jarque_bera_pvalue = jarque_bera(regression.wresid.values)[1]

    if jarque_bera_pvalue < .01:
        state.set("are_linear_model_regression_residuals_normally_distributed", -1)
        return -.5

    if jarque_bera_pvalue < .05:
        state.set(
            "are_linear_model_regression_residuals_normally_distributed", -0.5)
        return -.1

    if jarque_bera_pvalue < .1:
        state.set("are_linear_model_regression_residuals_normally_distributed", 0.5)
        return -.05

    state.set("are_linear_model_regression_residuals_normally_distributed", 1)
    return 0


def __penalty_for_correlation_of_error_terms(state: State, residuals: np.ndarray) -> float:
    dw_stat = durbin_watson(residuals)

    if 1 < dw_stat < 2:
        state.set("are_linear_model_regression_residuals_correlated", -1)
        return 0

    state.set("are_linear_model_regression_residuals_correlated", 1)
    return -.5


def __reward_for_homoscedasticity(state: State,
                                  residuals: np.ndarray,
                                  explanatory_vars: np.ndarray) -> float:
    f_stat_pvalue = het_breuschpagan(residuals, explanatory_vars)[3]

    if f_stat_pvalue < .01:
        state.set("are_linear_model_regression_residuals_heteroscedastic", 1)
        return -.5

    if f_stat_pvalue < .05:
        state.set("are_linear_model_regression_residuals_heteroscedastic", 0.5)
        return -.1

    if f_stat_pvalue < .1:
        state.set("are_linear_model_regression_residuals_heteroscedastic", -0.5)
        return 0.05

    state.set("are_linear_model_regression_residuals_heteroscedastic", -1)
    return 0


def __reward_for_recursive_residuals_mean(state: State,
                                          regression: RegressionResults) -> float:
    try:
        pvalue = linear_harvey_collier(regression)[1]
    except ValueError:
        state.set("is_linear_model_regression_recursive_residuals_mean_zero", -1)
        return -.5

    if pvalue < .01 or pvalue is nan:
        state.set("is_linear_model_regression_recursive_residuals_mean_zero", -1)
        return -.5

    if pvalue < .05:
        state.set("is_linear_model_regression_recursive_residuals_mean_zero", -0.5)
        return -.1

    if pvalue < .1:
        state.set("is_linear_model_regression_recursive_residuals_mean_zero", 0.5)
        return 0.05

    state.set("is_linear_model_regression_recursive_residuals_mean_zero", 1)
    return 0


linear_regression: Action[RegressionResults] = _linear_regression
