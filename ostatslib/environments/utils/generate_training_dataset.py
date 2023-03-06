"""
generate dataset function module
"""

from random import choice, randrange
from pandas import DataFrame
from scipy.stats import norm
from datacooker.recipes import LogitRecipe, PoissonRecipe, Recipe
from datacooker.variables import ContinousVariable


def generate_training_dataset() -> DataFrame:
    """
    Generates a dataset

    Returns:
       DataFrame: dataset
    """
    dataset_type = choice([LogitRecipe, PoissonRecipe, Recipe])
    recipe = __init_recipe(dataset_type)
    recipe.add_variable(ContinousVariable("a"))
    return recipe.cook(size=randrange(20, 1000))


def __init_recipe(dataset_type):
    if dataset_type == Recipe:
        recipe = dataset_type(lambda variables, error: 0 + 10 * variables["a"] + error)
        recipe.add_error(lambda variables, size: norm().rvs(size=size))
    else:
        recipe = dataset_type(lambda variables, _: 0 + variables["a"])
    return recipe
