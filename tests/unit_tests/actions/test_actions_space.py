import pytest
import numpy as np
from ostatslib.actions import ActionsSpace


def test_get_action_by_class_raises_exception_if_invalid_arg() -> None:
    """
    Test if get_action_by_class raises an exception when an invalid action type is passed
    """
    actions_space = ActionsSpace()
    with pytest.raises(ValueError):
        assert actions_space.get_action_by_class(type('NotAnAction', (), {}))


def test_get_action_accepts_integer_arg() -> None:
    """
    Test if get_action accepts int argument
    """
    actions_space = ActionsSpace()
    int_arg = 0
    ndarray_arg = np.zeros(actions_space.encoding_length)

    int_arg_action = actions_space.get_action(int_arg)
    ndarray_arg_action = actions_space.get_action(ndarray_arg)
    assert int_arg_action == ndarray_arg_action
