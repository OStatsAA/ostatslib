from random import choice, randrange
from pandas import DataFrame
from scipy.stats import norm
from datacooker.recipes import LogitRecipe, PoissonRecipe, Recipe
from datacooker.variables import ContinousVariable


def generate_training_datasets(datasets_count: int = 1000) -> list[tuple[str, DataFrame]]:
    """Generates a list of datasets

    Args:
        datasets_count (int, optional): datsets count. Defaults to 1000.

    Returns:
        list[tuple[str, DataFrame]]: list of datasets and their recipe type
    """
    datasets = [None] * datasets_count
    for index in range(datasets_count):
        dataset_type = choice([LogitRecipe, PoissonRecipe, Recipe])
        recipe = __init_recipe(dataset_type)
        recipe.add_variable(ContinousVariable("a"))
        datasets[index] = (dataset_type,
                           recipe.cook(size=randrange(20, 1000)))
    return datasets


def __init_recipe(dataset_type):
    if dataset_type == Recipe:
        recipe = dataset_type(lambda variables, error: 0 + 10 * variables["a"] + error)
        recipe.add_error(lambda variables, size: norm().rvs(size=size))
    else:
        recipe = dataset_type(lambda variables, _: 0 + variables["a"])
    return recipe
