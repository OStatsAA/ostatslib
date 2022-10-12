"""
Feature Extractor Testing
"""
import pytest
import pandas as pd

from ostatslib.features_extractor import FeaturesExtractor


@pytest.fixture
def fixture_dataframe() -> pd.DataFrame:
    """Reusable dataframe for testing functions

    Returns:
        Dataframe: dataframe
    """
    return pd.DataFrame({
        'V1': ['Row1', 'Row2', 'Row3'],
        'V2': [1, 1, 1],
        'V3': [True, False, True],
        'V4': [123, 123, 123]})


def test_should_have_total_rows_count(fixture_dataframe):
    """Testing total rows count"""
    features = FeaturesExtractor().run(fixture_dataframe, 'V4')

    assert features.rows_count == 3


def test_should_have_total_variables_count(fixture_dataframe):
    """Testing variables count"""
    features = FeaturesExtractor().run(fixture_dataframe, 'V4')

    assert features.variables_count == 4


def test_should_have_is_response_dichotomous_false(fixture_dataframe):
    """Testing total rows count"""
    features = FeaturesExtractor().run(fixture_dataframe, 'V4')

    assert features.is_response_dichotomous is False


def test_should_have_is_response_dichotomous_true(fixture_dataframe):
    """Testing total rows count"""
    features = FeaturesExtractor().run(fixture_dataframe, 'V3')

    assert features.is_response_dichotomous is True


def test_should_have_ratio_of_continous_variables(fixture_dataframe):
    """Testing total rows count"""
    features = FeaturesExtractor().run(fixture_dataframe, 'V4')

    assert features.ratio_of_continous_variables == .5
