"""split_x_y_data module
"""

from pandas import DataFrame, Series
from ostatslib.states import State


def split_x_y_data(data: DataFrame, state: State) -> tuple[DataFrame, Series]:
    """Splits DataFrame in X variables Dataframe and Y response data Series

    Args:
        data (DataFrame): data
        state (State): state

    Returns:
        tuple[DataFrame, Series]: X DataFrame, Y Series
    """
    target_label = state.get("response_variable_label")
    y_data = data[target_label]
    x_data = data.drop(target_label, axis=1)
    return x_data, y_data
