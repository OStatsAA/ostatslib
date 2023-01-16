"""
FeaturesSet abstract classe module
"""

from abc import ABC
from dataclasses import fields

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
