from numpy.lib.scimath import logn, log10
from numpy import tril, prod, isclose
from pandas import DataFrame
from ostatslib.actions.base import ExploratoryAction
from ostatslib.states import State


class LogColumnsCountExploration(ExploratoryAction):

    action_name = 'Log Columns Count'
    action_key = 'log_columns_count'
    _COUNT_UPPER_LIMIT = 1000

    def _explore(self, data: DataFrame, state: State) -> float:
        columns_count = len(data.columns)
        return min(logn(self._COUNT_UPPER_LIMIT, columns_count), 1)


class LogRowsCountExploration(ExploratoryAction):

    action_name = 'Log Rows Count'
    action_key = 'log_rows_count'
    _COMPRESSION_CONSTANT = 5.176
    """compression rate to keep log10(150K lines) close to 1"""

    def _explore(self, data: DataFrame, state: State) -> float:
        rows_count = len(data.index)
        return min(log10(rows_count)/self._COMPRESSION_CONSTANT, 1)


class CorrelatedVariablesRatioExploration(ExploratoryAction):

    action_name = 'Correlated Variables Ratio'
    action_key = 'correlated_variables_ratio'
    _CORRELATION_THRESHOLD = 0.5

    def _explore(self, data: DataFrame, state: State) -> float:
        corr_ratio = self.__get_correlated_ratio(data)
        return corr_ratio if corr_ratio else -1

    def __get_correlated_ratio(self, data: DataFrame) -> float:
        corr_matrix = data.corr()
        above_threshold_matrix = corr_matrix.abs() > self._CORRELATION_THRESHOLD
        return tril(above_threshold_matrix, -1).sum()/corr_matrix.shape[0]


class MissingDataRatioExploration(ExploratoryAction):

    action_name = 'Missing Data Ratio'
    action_key = 'missing_data_ratio'
    _OFFSET = 0.25

    def _explore(self, data: DataFrame, state: State) -> float:
        missing_count = data.isna().sum().sum()
        if not missing_count:
            return -1

        missing_ratio = (missing_count) / prod(data.shape)
        return min(missing_ratio + self._OFFSET, 1)


class StandardizedVariablesRatioExploration(ExploratoryAction):

    action_name = 'Standardized Variables Ratio'
    action_key = 'standardized_variables_ratio'

    def _explore(self, data: DataFrame, state: State) -> float:
        std_vars_ratio = self.__get_std_vars_ratio(data)
        return std_vars_ratio if std_vars_ratio else -1

    def __get_std_vars_ratio(self, data: DataFrame) -> float:
        data_stats = data.describe().loc[['mean', 'std']].T
        std_vars_filter = (isclose(data_stats['mean'], 0) &
                           isclose(data_stats['std'].round(2), 1))
        standardized_count = data_stats.loc[std_vars_filter]
        return standardized_count.shape[0] / data_stats.shape[0]
