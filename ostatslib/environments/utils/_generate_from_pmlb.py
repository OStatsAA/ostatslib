"""
Penn ML Benchmarks datasets module
"""

from random import choice
from pandas import DataFrame
import pmlb

from ostatslib.states import State

PMLB_CACHE_FOLDER = './.pmlb_cache'


def generate_from_pmlb() -> tuple[DataFrame, State]:
    """
    Randomly selects a dataset from Penn Machine Learning Benchmarks

    Returns:
        tuple[DataFrame, State]: dataset and initial state
    """
    pmlb_gen_fn = choice([
        __from_classification_datasets,
        __from_regression_datasets
    ])
    return pmlb_gen_fn()


def __from_classification_datasets() -> tuple[DataFrame, State]:
    dataset_name: str = choice(pmlb.classification_dataset_names)
    return __fetch(dataset_name)


def __from_regression_datasets() -> tuple[DataFrame, State]:
    dataset_name: str = choice(pmlb.regression_dataset_names)
    return __fetch(dataset_name)


def __fetch(dataset_name: str) -> tuple[DataFrame, State]:
    dataset = pmlb.fetch_data(dataset_name, local_cache_dir=PMLB_CACHE_FOLDER)
    if isinstance(dataset, DataFrame):
        state = State()
        state.set('response_variable_label', 'target')
        return dataset, state

    raise ValueError(f'Could not fetch {dataset_name} from PMLB')
