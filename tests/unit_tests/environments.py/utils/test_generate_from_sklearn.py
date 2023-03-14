# pylint: disable=protected-access
"""
GymEnvironment testing module
"""

from pandas import DataFrame
import ostatslib.environments.utils._generate_from_sklearn as sklearn_gen_module
from ostatslib.states import State


def test_from_toy_fn() -> None:
    """
    Tests each sklearn toy function
    """
    for toy_fn in sklearn_gen_module._TOY_FUNCTIONS:
        data, state = sklearn_gen_module._from_toy([toy_fn])
        assert isinstance(data, DataFrame)
        assert isinstance(state, State)
