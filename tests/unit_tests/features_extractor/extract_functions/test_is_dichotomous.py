"""Tests for is_quantitative function"""

import pandas as pd
import numpy as np

from ostatslib.features_extractor.extract_functions import is_dichotomous
from tests.utils import get_random_string


def test_should_flag_strings_series_as_false():
    str_series = pd.Series(get_random_string(5) for _ in range(100))
    assert is_dichotomous(str_series) is False


def test_should_flag_integers_series_as_false():
    int_series = pd.Series(np.random.randint(99, size=200))
    assert is_dichotomous(int_series) is False


def test_should_flag_dichotomous_integers_array_as_true():
    int_dichotomous_series = pd.Series(np.random.randint(1, 3, size=200))
    assert is_dichotomous(int_dichotomous_series) is True
