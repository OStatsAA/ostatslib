import operator
import numpy as np
from pandas import DataFrame, Series
from pandas.api.types import infer_dtype
from ostatslib.actions.base import TargetExploratoryAction
from ostatslib.actions.utils import split_x_y_data
from ostatslib.states import State


class ResponseUniqueValuesRatioExploration(TargetExploratoryAction):

    action_name = 'Response Unique Values Ratio'
    action_key = 'response_unique_values_ratio'

    def _explore(self, data: DataFrame, state: State) -> float:
        _, y_data = split_x_y_data(data, state)
        unique_ratio = y_data.nunique()/len(y_data)
        return unique_ratio


class InferResponseDTypeExploration(TargetExploratoryAction):

    action_name = 'Infer Response DType'
    action_key = 'response_inferred_dtype'

    def _explore(self, data: DataFrame, state: State) -> str:
        _, y_data = split_x_y_data(data, state)
        inferred_dtype = infer_dtype(y_data)
        return inferred_dtype


class IsResponseBalancedExploration(TargetExploratoryAction):

    action_name = 'Is Response Balanced'
    action_key = 'is_response_balanced'
    validations = [('response_unique_values_ratio', operator.lt, 0.25)]

    def _explore(self, data: DataFrame, state: State) -> float:
        _, y_data = split_x_y_data(data, state)
        unique_values_count = y_data.value_counts(normalize=True)
        min_max_counts_ratio = unique_values_count.min()/unique_values_count.max()
        if min_max_counts_ratio > 0.8:
            return 1
        if min_max_counts_ratio > 0.5:
            return 0.5
        if min_max_counts_ratio > 0.3:
            return -0.5

        return -1


class IsResponseDichotomousExploration(TargetExploratoryAction):

    action_name = 'Is Response Dichotomous'
    action_key = 'is_response_dichotomous'

    def _explore(self, data: DataFrame, state: State) -> int:
        _, y_data = split_x_y_data(data, state)
        return 1 if self.__is_dichotomous(y_data) else -1

    def __is_dichotomous(self, values: Series) -> bool:
        inferred_dtype = infer_dtype(values)
        if inferred_dtype == "boolean":
            return True

        unique_values = values.unique()
        if len(unique_values) > 2:
            return False

        match inferred_dtype:
            case "categorical" | "string" | "object" | "mixed-integer":
                return True
            case "integer":
                return self.__is_within_possible_boolean_range_of_integers(unique_values)
            case "floating" | "decimal" | "mixed-integer-float":
                first, second = unique_values
                return bool(first.is_integer() and
                            second.is_integer() and
                            self.__is_within_possible_boolean_range_of_integers(unique_values))
            case _:
                return False

    def __is_within_possible_boolean_range_of_integers(self, unique_values) -> bool:
        return bool(np.any((unique_values >= -1) | (unique_values <= 2)))


class IsResponseDiscreteExploration(TargetExploratoryAction):

    action_name = 'Is Response Discrete'
    action_key = 'is_response_discrete'

    def _explore(self, data: DataFrame, state: State) -> int:
        _, y_data = split_x_y_data(data, state)
        return 1 if self.__is_discrete(y_data) else -1

    def __is_discrete(self, values: Series) -> bool:
        unique_values = values.unique()

        is_numeric = np.issubdtype(unique_values.dtype, np.number)
        if not is_numeric:
            return False

        is_inexact = np.issubdtype(unique_values.dtype, np.inexact)
        if is_inexact:
            for value in unique_values:
                is_whole = float(value).is_integer()
                if not is_whole:
                    return False

        return True


class IsResponsePositiveValuesOnlyExploration(TargetExploratoryAction):

    action_name = 'Is Response Positive Values Only'
    action_key = 'is_response_positive_values_only'

    def _explore(self, data: DataFrame, state: State) -> int:
        _, y_data = split_x_y_data(data, state)
        return 1 if self.__is_positive_only_check(y_data) else -1

    def __is_positive_only_check(self, values: Series) -> bool:
        unique_values = values.unique()
        is_numeric = np.issubdtype(unique_values.dtype, np.number)

        if not is_numeric:
            try:
                unique_values.astype(float, copy=False)
            except Exception:
                return False

        return unique_values.min() >= 0


class IsResponseQuantitativeExploration(TargetExploratoryAction):

    action_name = 'Is Response Quantitative'
    action_key = 'is_response_quantitative'

    def _explore(self, data: DataFrame, state: State) -> int:
        _, y_data = split_x_y_data(data, state)
        return 1 if self.__is_quantitative_check(y_data) else -1

    def __is_quantitative_check(self, values: Series) -> bool:
        unique_values = values.unique()
        return np.issubdtype(unique_values.dtype, np.number)
