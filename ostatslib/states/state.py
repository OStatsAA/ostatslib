"""
State abstract class module
"""

from abc import ABC
from dataclasses import fields
from numpy import NaN, concatenate, isnan, ndarray, array
from ostatslib.features_extractors import AnalysisFeaturesSet, DataFeaturesSet
from ostatslib.states.state_iterator import StateIterator


class State(ABC):
    """
    State abstract class
    """

    def __init__(self,
                 data_features: DataFeaturesSet,
                 analysis_features: AnalysisFeaturesSet) -> None:
        self.__data_features: DataFeaturesSet = data_features
        self.__analysis_features: AnalysisFeaturesSet = analysis_features

    def get(self, feature_key: str) -> int | float | bool:
        """
        Gets feature value by passing feature key (feature name).
        If feature is not found, returns NaN.

        Args:
            feature_key (str): feature key (name)

        Raises:
            AttributeError: raises error if feature is not found

        Returns:
            int | float | bool: feature value
        """
        value = getattr(self.__data_features, feature_key, NaN)
        if isinstance(value, dict) or not isnan(value):
            return value

        value = getattr(self.__analysis_features, feature_key, NaN)
        if isinstance(value, str) or not isnan(value):
            return value

        raise AttributeError()

    def set(self, feature_key: str, value: int | float | bool) -> None:
        """
        Sets value to feature

        Args:
            feature_key (str): feature key (name)
            value (int | float | bool): value

        Raises:
            AttributeError: If feature is not found
        """
        if hasattr(self.__data_features, feature_key):
            setattr(self.__data_features, feature_key, value)
        elif hasattr(self.__analysis_features, feature_key):
            setattr(self.__analysis_features, feature_key, value)
        else:
            raise AttributeError()

    def keys(self) -> list[str]:
        """
        Returns features keys (names)

        Returns:
            list[str]: list of features keys (names)
        """
        features_fields = fields(self.__data_features)
        features_fields += fields(self.__analysis_features)
        return list(map(lambda field: field.name, features_fields))

    @property
    def features_vector(self) -> ndarray[float]:
        """
        Returns features vector

        Returns:
            ndarray: array of values
        """
        return concatenate((
            array(self.__analysis_features),
            array(self.__data_features)
        ))

    def __iter__(self):
        return StateIterator(self)

    def __eq__(self, other) -> bool:
        return self is other or self.__check_if_features_match(other)

    def __check_if_features_match(self, other) -> bool:
        for field in fields(self.__data_features):
            if getattr(self.__data_features, field.name) != other.get(field.name):
                return False

        for field in fields(self.__analysis_features):
            if getattr(self.__analysis_features, field.name) != other.get(field.name):
                return False

        return True
