"""
ActionsSpace testing module
"""

from ostatslib.actions import ActionsSpace


def test_len_method() -> None:
    """
    Length of an ActionsSpace is the length of its internal actions dictionary
    """
    actions_space = ActionsSpace()
    actions_dict = actions_space.actions

    assert len(actions_space) == len(actions_dict)


def test_actions_space_get_action_returns_a_function() -> None:
    """
    ActionsSpace actions dict should have functions in values
    """
    actions_space = ActionsSpace()
    for action in actions_space.actions.values():
        assert callable(action)


def test_actions_space_should_have_method_to_get_action_fn_by_name() -> None:
    """
    ActionsSpace should have a method to get action functions by name
    """
    actions_space = ActionsSpace()
    action = actions_space.get_action('linear_regression')

    assert callable(action)


def test_actions_space_should_have_method_to_get_action_names_list() -> None:
    """
    ActionsSpace exposes actions names list
    """
    actions_space = ActionsSpace()
    actions_names = actions_space.actions_names_list

    assert isinstance(actions_names, list)
