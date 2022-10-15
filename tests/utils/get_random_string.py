"""
get_random_string function module
"""

import random
import string


def get_random_string(length):
    """Generate a random string

    Args:
        length (int): length of the generated string

    Returns:
        str: Randomly generated string.
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))
