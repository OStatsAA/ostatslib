"""
FeaturesSet abstract classe module
"""

from abc import ABC
from dataclasses import Field, fields
import numpy as np

KnownFeaturesList = list[tuple[str, float | int | str]]


class FeaturesSet(ABC):
    """
    Abstract base class for FeaturesSets
    """

    def list_known_features(self) -> KnownFeaturesList:
        """
        Lists fields that have values different from default (unkown state attribute)

        Returns:
            KnownFeaturesList: list of non-default values
        """
        known_features = []
        for field in fields(self):
            value = getattr(self, field.name)
            if field.default != value:
                known_features.append((field.name, value))

        return known_features

    def as_gymnasium_space(self) -> dict:
        """
        Features as Gymnasium space

        Returns:
            dict: dictionary of features and their Gymnasium spaces
        """
        return dict((field.name, field.metadata['gym_space']) for field in fields(self))

    def as_features_dict(self) -> dict:
        """
        Features values as dictionary

        Returns:
            dict: dictionary with features values
        """
        keys = [field.name for field in fields(self)]
        values = self.__array__()
        return dict(zip(keys, values))

    def __array__(self):
        return np.array([self.__get_feature_value(field) for field in fields(self)])

    def __get_feature_value(self, _field: Field) -> str | float | int:
        get_value_fn = _field.metadata['get_value_fn']
        value = getattr(self, _field.name)
        if get_value_fn is None:
            return np.array(value).reshape((1,))

        return np.array(get_value_fn(value)).reshape((1,))
