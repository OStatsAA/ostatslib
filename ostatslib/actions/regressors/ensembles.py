import operator
from sklearn.ensemble import (AdaBoostRegressor,
                              BaggingRegressor,
                              ExtraTreesRegressor,
                              GradientBoostingRegressor,
                              RandomForestRegressor)

from ostatslib.actions.base import TargetModelEstimatorAction, TreeEstimatorAction


class AdaBoostRegression(TargetModelEstimatorAction[AdaBoostRegressor]):

    action_name = 'AdaBoost'
    action_key = 'adaboost_regression'
    estimator = AdaBoostRegressor()
    params_grid = {'n_estimators': [10, 50, 100],
                   'loss': ['linear', 'square', 'exponential']}
    exceptions_handlers = None
    validations = [('response_unique_values_ratio', operator.gt, 0.1)]


class BaggingRegression(TargetModelEstimatorAction[BaggingRegressor]):

    action_name = 'Bagging'
    action_key = 'bagging_regression'
    estimator = BaggingRegressor()
    params_grid = {'n_estimators': [5, 10, 25]}
    exceptions_handlers = None
    validations = [('response_unique_values_ratio', operator.gt, 0.1)]


class ExtraTreesRegression(TreeEstimatorAction[ExtraTreesRegressor]):

    action_name = 'Extra-Trees'
    action_key = 'extra_trees_regression'
    estimator = ExtraTreesRegressor()
    params_grid = {'n_estimators': [10, 50, 100],
                   'criterion': ['squared_error', 'friedman_mse'],
                   'max_features': ['sqrt', 'log2'],
                   'ccp_alpha': [1e-2, 1e-3]}
    exceptions_handlers = None
    validations = [('response_unique_values_ratio', operator.gt, 0.1)]


class GradientBoostingRegression(TargetModelEstimatorAction[GradientBoostingRegressor]):

    action_name = 'Gradient Boosting'
    action_key = 'gradient_boosting_regression'
    estimator = GradientBoostingRegressor()
    params_grid = {'n_estimators': [10, 50],
                   'loss': ['squared_error', 'huber', 'quantile'],
                   'criterion': ['friedman_mse', 'squared_error'],
                   'max_features': ['sqrt', 'log2'],
                   'ccp_alpha': [1e-2, 1e-3]}
    exceptions_handlers = None
    validations = [('response_unique_values_ratio', operator.gt, 0.1)]


class N100EstimatorsGradientBoostingRegression(TargetModelEstimatorAction[GradientBoostingRegressor]):

    action_name = 'Gradient Boosting 100 Estimators'
    action_key = 'n_100_estimators_gradient_boosting_regression'
    estimator = GradientBoostingRegressor(n_estimators=100)
    params_grid = {'criterion': ['friedman_mse', 'squared_error'],
                   'max_features': ['sqrt', 'log2'],
                   'ccp_alpha': [1e-2, 1e-3]}
    exceptions_handlers = None
    validations = [('response_unique_values_ratio', operator.gt, 0.1)]


class RandomForestRegression(TreeEstimatorAction[RandomForestRegressor]):

    action_name = 'Random Forest'
    action_key = 'random_forest_regression'
    estimator = RandomForestRegressor()
    params_grid = {'n_estimators': [10, 50],
                   'criterion': ['squared_error', 'friedman_mse'],
                   'max_features': ['sqrt', 'log2'],
                   'ccp_alpha': [1e-2, 1e-3]}
    exceptions_handlers = None
    validations = [('response_unique_values_ratio', operator.gt, 0.1)]


class N100EstimatorsRandomForestRegression(TreeEstimatorAction[RandomForestRegressor]):

    action_name = 'Random Forest 100 Estimators'
    action_key = 'n_100_estimators_random_forest_regression'
    estimator = RandomForestRegressor(n_estimators=100)
    params_grid = {'max_features': ['sqrt', 'log2'],
                   'ccp_alpha': [1e-2, 1e-3]}
    exceptions_handlers = None
    validations = [('response_unique_values_ratio', operator.gt, 0.1)]
