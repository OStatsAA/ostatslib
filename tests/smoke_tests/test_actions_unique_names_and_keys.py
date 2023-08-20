from ostatslib.actions import ActionsSpace


def test_actions_unique_names() -> None:
    actions_space = ActionsSpace()
    actions_names = [
        action.action_name for action in actions_space.actions_list]
    assert len(actions_names) == len(set(actions_names))


def test_actions_unique_keys() -> None:
    actions_space = ActionsSpace()
    actions_keys = [action.action_key for action in actions_space.actions_list]
    assert len(actions_keys) == len(set(actions_keys))
