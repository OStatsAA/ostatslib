"""Tests for is_quantitative function"""

import pandas as pd
import numpy as np

from ostatslib.features_extractor.extract_functions import is_quantitative
from ostatslib.features_extractor.extract_functions.is_quantitative import MEDIUM_ROW_COUNT
from tests.utils import get_random_string


def test_should_flag_strings_series_as_false():
    str_series = pd.Series(get_random_string(5) for _ in range(100))
    assert is_quantitative(str_series) is False


def test_should_flag_integers_series_as_true():
    int_series = pd.Series(np.random.randint(99, size=200))
    assert is_quantitative(int_series) is True


def test_should_flag_dichotomous_integers_series_as_false():
    int_dichotomous_series = pd.Series(np.random.randint(1, 3, size=200))
    assert is_quantitative(int_dichotomous_series) is False


def test_should_flag_dichotomous_floating_series_as_false():
    float_dichotomous_series = pd.Series(np.random.choice([.5, 1.5], size=200))
    assert is_quantitative(float_dichotomous_series) is False


def test_should_flag_low_unique_count_as_false():
    """Only a few floating options indicates variable may be categorical
    """
    float_series = pd.Series(np.random.choice(
        [.5, 1.5, 2.5], size=MEDIUM_ROW_COUNT + 1))
    assert is_quantitative(float_series) is False


def test_should_flag_low_unique_count_in_a_medium_dataset_as_false():
    """Only a few floating options should be continous in tiny datasets
    """
    int_series = pd.Series(
        np.random.randint(1, 4, size=MEDIUM_ROW_COUNT + 1))
    assert is_quantitative(int_series) is False


def test_should_flag_low_unique_count_in_a_tiny_dataset_as_true():
    """Only a few floating options should be continous in tiny datasets
    """
    float_series = pd.Series(np.random.choice(
        [.5, 1.5, 2.5], size=MEDIUM_ROW_COUNT - 1))
    assert is_quantitative(float_series) is True
