from pandas import DataFrame
from random import randrange
from scipy.stats import norm
from datacooker.recipes import LogitRecipe, Recipe
from datacooker.variables import ContinousVariable


def generate_training_datasets(datasets_count: int = 1000) -> list[DataFrame]:
    datasets = [None] * datasets_count
    for index in range(datasets_count):
        recipe = None
        if index % 2:
            recipe = Recipe(lambda variables, error: 0 +
                            10 * variables["a"] + error)
            recipe.add_error(lambda variables, size: norm().rvs(size=size))
        else:
            recipe = LogitRecipe(lambda variables, _: 0 + 10 * variables["a"])
        recipe.add_variable(ContinousVariable("a"))
        datasets[index] = recipe.cook(size=randrange(20, 2000))
    return datasets
