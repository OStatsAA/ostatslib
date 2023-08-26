import operator
from datasetsforecast.losses import rmse
from numpy import ndarray
from pandas import DataFrame, Series, infer_freq
from sklearn.linear_model import GammaRegressor, LinearRegression, PoissonRegressor
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsmodels.tools.tools import add_constant
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan

from ostatslib.actions.base import ActionInfo, TargetModelEstimatorAction
from ostatslib.actions.utils import split_x_y_data
from ostatslib.config import Config
from ostatslib.states import State


class AutoARIMARegression(TargetModelEstimatorAction[AutoARIMA]):
    action_name = 'AutoARIMA Regression'
    action_key = 'auto_arima_regression'
    estimator = AutoARIMA()
    validations = [('time_convertible_variable', operator.truth, None)]

    def _fit(self, data: DataFrame, state: State) -> tuple[AutoARIMA, float]:
        time_var_label = self.__get_time_var_label(state)
        freq = self.__get_frequency(data, time_var_label)

        x_data, y_data = split_x_y_data(data, state)
        x_data.drop(time_var_label, axis=1, inplace=True)

        stats_forecast_df = self.__build_stats_forecast_dataframe(data,
                                                                  time_var_label,
                                                                  y_data)

        stats_forecast = StatsForecast(df=stats_forecast_df,
                                       models=[self.estimator],
                                       freq=freq).fit()
        score = self.__get_score(stats_forecast, y_data)
        return stats_forecast.fitted_[0, 0], score

    def __build_stats_forecast_dataframe(self,
                                         data: DataFrame,
                                         time_var_label: str,
                                         y_data: Series) -> DataFrame:
        return DataFrame({'unique_id': ['id'] * len(y_data),
                          'ds': data[time_var_label],
                          'y': y_data})

    def __get_time_var_label(self, state: State) -> str:
        time_var_label: str | None = state.get('time_convertible_variable')
        if time_var_label:
            return time_var_label

        raise ValueError('Not time variable assigned in state')

    def __get_frequency(self, data: DataFrame, time_var_label: str) -> str:
        freq: str | None = infer_freq(data[time_var_label])
        if freq:
            return freq

        raise ValueError(f'Cannot infer frequency of {time_var_label}')

    def __get_score(self, stats_forecast: StatsForecast, y_data: Series) -> float:
        cv_data = stats_forecast.cross_validation(len(y_data) // 5)
        rmse_: float = rmse(cv_data['y'], cv_data['AutoARIMA'])
        return 1 - (rmse_/cv_data['y'].mean())


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
    pvalue = jarque_bera(residuals)[1]
    return _update_state_and_get_diagnostic_reward(state, config, pvalue, feature_key)


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
    feature_key = 'are_linear_model_regression_residuals_homoscedastic'
    pvalue = het_breuschpagan(residuals, add_constant(x_data))[3]
    return _update_state_and_get_diagnostic_reward(state, config, pvalue, feature_key)


def _update_state_and_get_diagnostic_reward(state: State,
                                            config: Config,
                                            pvalue: float,
                                            feature_key: str) -> float:
    if pvalue < config['FULL_PENALIZED_PVALUE']:
        state.set(feature_key, -1)
        return -.5

    if pvalue < config['PARTIAL_PENALIZED_PVALUE']:
        state.set(feature_key, -0.5)
        return -.1

    state.set(feature_key, 1)
    return 0
