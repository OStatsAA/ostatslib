# pylint: disable=redefined-outer-name
# pylint: disable=broad-except
"""
State testing module
"""

from copy import deepcopy
import pytest
from ostatslib.features_extractors import AnalysisFeaturesSet, DataFeaturesSet
from ostatslib.states import State

NON_EXISTING_KEY = "NON_EXISTING_KEY_IN_STATE_FEATURES_BLABLABLA"
KNOWN_FEATURE_KEY = "is_response_dichotomous"


@pytest.fixture
def dummy_state() -> State:
    """
    Instantiates a dummy state fixture
    """
    return State(DataFeaturesSet(), AnalysisFeaturesSet())


def test_state_gets_method(dummy_state) -> None:
    """
    Tests if state.get() method is able to return all keys values
    """
    state: State = dummy_state
    try:
        for key in state.keys():
            state.get(key)
    except Exception as exception:
        assert False, f"getting and existing key raised an exception {exception}"


def test_state_gets_method_raises_attribute_error_for_invalid_keys(dummy_state) -> None:
    """
    Tests if state.get() method raises AtributeError for invalid features keys
    """
    state: State = dummy_state
    with pytest.raises(AttributeError):
        assert state.get(NON_EXISTING_KEY)


def test_state_sets_method(dummy_state) -> None:
    """
    Tests if state.set() updates values for a feature by key
    """
    state: State = dummy_state

    value = state.get(KNOWN_FEATURE_KEY)
    state.set(KNOWN_FEATURE_KEY, not value)

    updated_value = state.get(KNOWN_FEATURE_KEY)

    assert value != updated_value


def test_state_sets_method_raises_attribute_error_for_invalid_keys(dummy_state) -> None:
    """
    Tests if state.set() method raises AtributeError for invalid features keys
    """
    state: State = dummy_state
    with pytest.raises(AttributeError):
        assert state.set(NON_EXISTING_KEY, 42)


def test_state_iterator(dummy_state) -> None:
    """
    Tests if there's a custom iterator implemented for State objects that return (key, value) tuples
    """
    state: State = dummy_state
    for key, value in state:
        assert value == state.get(key)


def test_state_equality_operator(dummy_state) -> None:
    """
    If a State variable refers to the same object or have the same features values, \
    equality operation should return True
    """
    state: State = dummy_state
    other_state_ref = state
    assert state is other_state_ref

    other_state = deepcopy(state)
    assert state is not other_state
    assert state == other_state

    other_state.set(KNOWN_FEATURE_KEY, not state.get(KNOWN_FEATURE_KEY))
    assert state != other_state


def test_state_should_expose_features_vector(dummy_state: State) -> None:
    """
    Tests if state is able to expose its features as a vector
    """
    state: State = dummy_state
    state.set("score", .5)

    assert state.features_vector[0] == .5
