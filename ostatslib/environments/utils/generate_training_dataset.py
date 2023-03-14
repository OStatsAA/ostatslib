"""
generate dataset function module
"""

from random import choice
from pandas import DataFrame

from ostatslib.states import State
from ._generate_from_datacooker import generate_from_datacooker
from ._generate_from_sklearn import generate_from_sklearn


def generate_training_dataset() -> tuple[DataFrame, State]:
    """
    Generates a dataset

    Returns:
       DataFrame: dataset
    """
    generator_fn = choice([generate_from_datacooker, generate_from_sklearn])
    return generator_fn()
