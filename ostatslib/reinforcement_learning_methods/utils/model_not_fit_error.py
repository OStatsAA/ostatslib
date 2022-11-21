"""
Model not fit error module
"""


class ModelNotFitError(Exception):
    """
    Exception raised if model has not been fitted
    """

    def __init__(self, message="QValue aproximation model has not been fitted"):
        self.message = message
        super().__init__(self.message)
