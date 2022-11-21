"""
convert state to tensor module
"""

from tensorflow import Tensor, convert_to_tensor, expand_dims

from ostatslib.states import State


def convert_state_to_tensor(state: State) -> Tensor:
    """
    Converts OStatsLib State to Tensorflow Tensor

    Args:
        state (State): state

    Returns:
        Tensor: Tensorflow Tensor
    """
    state_tensor = convert_to_tensor(state.features_vector)
    state_tensor: Tensor = expand_dims(state_tensor, 0)
    return state_tensor
