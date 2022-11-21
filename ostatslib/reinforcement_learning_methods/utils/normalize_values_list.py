"""
normalize list values module
"""

import numpy as np

# Smallest number such that 1.0 + eps != 1.0
__EPS = np.finfo(np.float32).eps.item()


def normalize_values_list(values_list: list[int | float]) -> list[int | float]:
    """
    Normalizes values in a list

    Args:
        values_list (list[int | float]): values list

    Returns:
        list[int | float]: normalized values list
    """
    values_list = np.array(values_list)
    values_list = (values_list - np.mean(values_list)) / \
        (np.std(values_list) + __EPS)
    values_list = values_list.tolist()
    return values_list
