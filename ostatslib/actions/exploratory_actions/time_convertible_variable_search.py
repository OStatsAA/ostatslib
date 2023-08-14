from pandas import DataFrame
from pandas.api.types import infer_dtype
from ostatslib.actions.base import ExploratoryAction
from ostatslib.actions.utils import split_x_y_data
from ostatslib.states import State


class TimeConvertibleVariableSearch(ExploratoryAction):

    action_name = 'Time Convertible Variable Search'
    action_key = 'time_convertible_variable'
    _ORDERED_LIST_OF_TIME_DTYPES = [
        "datetime64", "datetime", "date",
        "period",
        "time",
        "timedelta64", "timedelta",
    ]

    def _explore(self, data: DataFrame, state: State) -> str | None:
        x_data, _ = split_x_y_data(data, state)
        time_vars: list[tuple[str, str]] = []

        for var_name, values in x_data.items():
            inferred_dtype: str = infer_dtype(values)
            if inferred_dtype in self._ORDERED_LIST_OF_TIME_DTYPES:
                time_vars.append((var_name, inferred_dtype))

        if not time_vars:
            return None

        if len(time_vars) == 1:
            return time_vars[0][0]

        return self.__select_best_time_related_variable(time_vars)

    def __select_best_time_related_variable(self, time_related_variables) -> str:
        for time_dtype in self._ORDERED_LIST_OF_TIME_DTYPES:
            for var_name, inferred_dtype_name in time_related_variables:
                if time_dtype == inferred_dtype_name:
                    return var_name

        return time_related_variables[0]
