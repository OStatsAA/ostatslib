"""Tests for is_quantitative function"""

import pandas as pd
import numpy as np

from ostatslib.features_extractors.data_features_extractor.extract_functions import get_missing_entries_ratio
from tests.utils import get_random_string


def test_should_flag_empty_strings_as_missing_entries():
    str_series = pd.Series(get_random_string(5) for _ in range(100))
    expected_ratio = .25
    str_series.loc[str_series.sample(frac=expected_ratio).index] = ""
    ratio = get_missing_entries_ratio(str_series.to_numpy())
    assert ratio == expected_ratio


def test_should_flag_nan_as_missing_entries_in_numeric_array():
    int_series = pd.Series(np.random.randint(99, size=200))
    expected_ratio = .25
    int_series.loc[int_series.sample(frac=expected_ratio).index] = np.NaN
    ratio = get_missing_entries_ratio(int_series.to_numpy())
    assert ratio == expected_ratio
