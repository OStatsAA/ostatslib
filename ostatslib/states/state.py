"""
State module
"""

from dataclasses import fields
from numpy import concatenate, ndarray, array
from ostatslib.states.analysis_features_set import AnalysisFeaturesSet
from ostatslib.states.data_features_set import DataFeaturesSet
from ostatslib.states.features_set import KnownFeaturesList
from ostatslib.states.state_iterator import StateIterator


class State:
    """
    State class
    """

    def __init__(self,
                 data_features: DataFeaturesSet = None,
                 analysis_features: AnalysisFeaturesSet = None) -> None:
        self.__data_features = (
            data_features if data_features is not None else DataFeaturesSet())
        self.__analysis_features = (
            analysis_features if analysis_features is not None else AnalysisFeaturesSet())

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
        if hasattr(self.__data_features, feature_key):
            return getattr(self.__data_features, feature_key)

        if hasattr(self.__analysis_features, feature_key):
            return getattr(self.__analysis_features, feature_key)

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
            return setattr(self.__data_features, feature_key, value)

        if hasattr(self.__analysis_features, feature_key):
            return setattr(self.__analysis_features, feature_key, value)

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

    def list_known_features(self) -> KnownFeaturesList:
        """
        Lists fields that have values different from default (unkown state attribute)

        Returns:
            KnownFeaturesList: list of non-default values
        """
        return [
            *self.__analysis_features.list_known_features(),
            *self.__data_features.list_known_features()
        ]

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

    def __sub__(self, other):
        diff_state: State = State()
        sub_sets = [self.__analysis_features, self.__data_features]
        for sub_set in sub_sets:
            for field in fields(sub_set):
                value = getattr(sub_set, field.name)
                if value != other.get(field.name):
                    diff_state.set(field.name, value)

        return diff_state
