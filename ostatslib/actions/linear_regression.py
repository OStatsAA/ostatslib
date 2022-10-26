"""
Linear regression module

ref:
https://www.kirenz.com/post/2021-11-14-linear-regression-diagnostics-in-python/linear-regression-diagnostics-in-python/
"""

import numpy as np
from pandas import DataFrame
from scipy.stats import kstest, norm
from statsmodels.api import OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.regression.linear_model import RegressionResults

from ostatslib.actions.utils import ActionResult, reward_cap, interpretable_model
from ostatslib.states import State


@reward_cap
@interpretable_model
def linear_regression(state: State,
                      data: DataFrame) -> ActionResult[RegressionResults]:
    """
    Fits data to a linear regression model.

    Args:
        state (State): current environment state
        data (DataFrame): data under analysis

    Returns:
        ActionResult[RegressionResults]: _description_
    """
    response_var_label = state.get("response_variable_label")
    response_var = data[response_var_label]
    explanatory_vars = data.drop(response_var_label, axis=1)
    regression = OLS(response_var, explanatory_vars).fit()
    reward = __calculate_reward(state, regression)
    return ActionResult(state, reward, regression)


def __calculate_reward(state: State, regression: RegressionResults) -> float:
    explanatory_vars: np.ndarray = regression.model.exog
    residuals: np.ndarray = regression.resid.values
    residuals_studentized: np.ndarray = regression.get_influence().resid_studentized
    reward: float = 0

    reward += __penalty_for_dichotomous_response(state)
    reward += __reward_for_normally_distributed_errors(residuals_studentized)
    reward += __reward_for_correlation_of_error_terms(residuals)
    reward += __reward_for_homoscedasticity(residuals, explanatory_vars)

    return reward


def __penalty_for_dichotomous_response(state: State) -> State:
    if state.get("is_response_dichotomous") > 0:
        return -50

    return 0


def __reward_for_normally_distributed_errors(residuals_studentized: np.ndarray) -> float:
    std_residuals_goodness_of_fit = kstest(residuals_studentized,
                                           norm(0, 1).cdf)

    if std_residuals_goodness_of_fit.pvalue < .01:
        return -20

    if std_residuals_goodness_of_fit.pvalue < .05:
        return -10

    return 10


def __reward_for_correlation_of_error_terms(residuals: np.ndarray) -> float:
    dw_stat = durbin_watson(residuals)

    if 1 < dw_stat < 2:
        return 10

    return -10


def __reward_for_homoscedasticity(residuals: np.ndarray, explanatory_vars: np.ndarray) -> float:
    f_stat_pvalue = het_breuschpagan(residuals, explanatory_vars)[3]

    if f_stat_pvalue < .01:
        return -20

    if f_stat_pvalue < .05:
        return -10

    return 10
