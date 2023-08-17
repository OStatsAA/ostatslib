from numpy import ndarray
from pandas import DataFrame
from sklearn.linear_model import GammaRegressor, LinearRegression, PoissonRegressor
from statsmodels.tools.tools import add_constant
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan

from ostatslib.actions.base import ActionInfo, TargetModelEstimatorAction
from ostatslib.actions.utils import split_x_y_data
from ostatslib.config import Config
from ostatslib.states import State


class PoissonRegression(TargetModelEstimatorAction[PoissonRegressor]):

    action_name = 'Poisson Regression'
    action_key = 'poisson_regression'
    estimator = PoissonRegressor()


class GammaRegression(TargetModelEstimatorAction[GammaRegressor]):

    action_name = 'Gamma Regression'
    action_key = 'gamma_regression'
    estimator = GammaRegressor()


class OLSLinearRegression(TargetModelEstimatorAction[LinearRegression]):

    action_name = 'Linear Regression'
    action_key = 'linear_regression'
    estimator = LinearRegression()

    def execute(self,
                data: DataFrame,
                state: State,
                config: Config) -> tuple[State, float, ActionInfo[LinearRegression]]:
        state, reward, info = super().execute(data, state, config)
        if not isinstance(info.model, LinearRegression):
            return state, reward, info

        state, reward = self.__update_reward_state_diagnostics(data,
                                                               state,
                                                               config,
                                                               reward,
                                                               info.model)
        reward = min(max(config['MIN_REWARD'], reward), config['MAX_REWARD'])
        info.next_state = state.copy()
        return state, reward, info

    def __update_reward_state_diagnostics(self,
                                          data: DataFrame,
                                          state: State,
                                          config: Config,
                                          reward: float,
                                          model: LinearRegression) -> tuple[State, float]:
        x_data, y_data = split_x_y_data(data, state)
        residuals: ndarray = y_data.values - model.predict(x_data)

        reward += _reward_for_normally_distributed_errors(state,
                                                           residuals,
                                                           config)
        reward += _penalty_for_correlation_of_error_terms(state, residuals)
        reward += _reward_for_homoscedasticity(state,
                                                residuals,
                                                x_data.values,
                                                config)

        return state, reward


def _reward_for_normally_distributed_errors(state: State,
                                             residuals: ndarray,
                                             config: Config) -> float:
    feature_key = 'are_linear_model_regression_residuals_normally_distributed'
    jarque_bera_pvalue = jarque_bera(residuals)[1]

    if jarque_bera_pvalue < config['FULL_PENALIZED_PVALUE']:
        state.set(feature_key, -1)
        return -.5

    if jarque_bera_pvalue < config['PARTIAL_PENALIZED_PVALUE']:
        state.set(feature_key, -0.5)
        return -.1

    if jarque_bera_pvalue < .1:
        state.set(feature_key, 0.5)
        return -.05

    state.set(feature_key, 1)
    return 0


def _penalty_for_correlation_of_error_terms(state: State, residuals: ndarray) -> float:
    feature_key = 'are_linear_model_regression_residuals_correlated'
    dw_stat = durbin_watson(residuals)

    if 1 < dw_stat < 2:
        state.set(feature_key, -1)
        return 0

    state.set(feature_key, 1)
    return -.5


def _reward_for_homoscedasticity(state: State,
                                  residuals: ndarray,
                                  x_data: ndarray,
                                  config: Config) -> float:
    feature_key = 'are_linear_model_regression_residuals_heteroscedastic'
    f_stat_pvalue = het_breuschpagan(residuals, add_constant(x_data))[3]

    if f_stat_pvalue < config['FULL_PENALIZED_PVALUE']:
        state.set(feature_key, 1)
        return -.5

    if f_stat_pvalue < config['PARTIAL_PENALIZED_PVALUE']:
        state.set(feature_key, 0.5)
        return -.1

    if f_stat_pvalue < .1:
        state.set(feature_key, -0.5)
        return 0.05

    state.set(feature_key, -1)
    return 0
