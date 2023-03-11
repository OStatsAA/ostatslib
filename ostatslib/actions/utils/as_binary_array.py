"""
as_binary_array helper function module
"""

from numpy import ndarray, array


def as_binary_array(number: int, array_length: int) -> ndarray:
    """
    Converts an integer to a np.array of binary digits

    Args:
        number (int): integer number to be converted

    Returns:
        array: numpy array of converted digits
    """
    binary_str = bin(number)[2:].zfill(array_length)

    return array([int(digit) for digit in binary_str])
