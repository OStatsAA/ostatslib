"""Ensembles regression actions module
"""

import operator
from sklearn.ensemble import (AdaBoostRegressor,
                              BaggingRegressor,
                              ExtraTreesRegressor,
                              RandomForestRegressor)
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from ostatslib.actions.base import TargetModelEstimatorAction, TreeEstimatorAction


class AdaBoostRegression(TargetModelEstimatorAction[AdaBoostRegressor]):
    """AdaBoost regression action with linear loss.
    Fits a Scikit-Learn AdaBoostRegressor
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
    """

    action_name = 'AdaBoost Regression'
    action_key = 'adaboost_regression'
    estimator = AdaBoostRegressor(loss='linear')
    params_grid = {'n_estimators': [10, 50, 100]}
    exceptions_handlers = None
    validations = [('response_unique_values_ratio', operator.gt, 0.1)]


class AdaBoostSquareLossRegression(TargetModelEstimatorAction[AdaBoostRegressor]):
    """AdaBoost regression action with square loss.
    Fits a Scikit-Learn AdaBoostRegressor
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
    """

    action_name = 'AdaBoost Regression with Square Loss'
    action_key = 'adaboost_square_loss_regression'
    estimator = AdaBoostRegressor(loss='square')
    params_grid = {'n_estimators': [10, 50, 100]}
    exceptions_handlers = None
    validations = [('response_unique_values_ratio', operator.gt, 0.1)]


class AdaBoostExponentialLossRegression(TargetModelEstimatorAction[AdaBoostRegressor]):
    """AdaBoost regression action with exponential loss.
    Fits a Scikit-Learn AdaBoostRegressor
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
    """

    action_name = 'AdaBoost Regression with Exponential Loss'
    action_key = 'adaboost_exponential_loss_regression'
    estimator = AdaBoostRegressor(loss='exponential')
    params_grid = {'n_estimators': [10, 50, 100]}
    exceptions_handlers = None
    validations = [('response_unique_values_ratio', operator.gt, 0.1)]


class BaggingRegression(TargetModelEstimatorAction[BaggingRegressor]):
    """Bagging regression action.
    Fits a Scikit-Learn BaggingRegressor
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html
    """

    action_name = 'Bagging Regression'
    action_key = 'bagging_regression'
    estimator = BaggingRegressor()
    params_grid = {'estimator': [DecisionTreeRegressor(max_depth=3),
                                 DecisionTreeRegressor(max_depth=10)],
                   'n_estimators': [5, 10, 25]}
    exceptions_handlers = None
    validations = [('response_unique_values_ratio', operator.gt, 0.1)]


class ExtraTreesRegression(TreeEstimatorAction[ExtraTreesRegressor]):
    """ExtraTrees regression action.
    Fits a Scikit-Learn ExtraTreesRegressor
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
    """

    action_name = 'Extra-Trees Regression'
    action_key = 'extra_trees_regression'
    estimator = ExtraTreesRegressor()
    params_grid = {'n_estimators': [10, 50, 100],
                   'criterion': ['squared_error', 'friedman_mse'],
                   'max_features': ['sqrt', 'log2']}
    exceptions_handlers = None
    validations = [('response_unique_values_ratio', operator.gt, 0.1)]


class GradientBoostingRegression(TreeEstimatorAction[XGBRegressor]):
    """Gradient Boosting regression action.
    Fits a XGBoostClassfier
    https://xgboost.readthedocs.io/en/stable/python/examples/basic_walkthrough.html
    """

    action_name = 'Gradient Boosting Regression'
    action_key = 'gradient_boosting_regression'
    estimator = XGBRegressor()
    params_grid = {'n_estimators': [10, 50]}
    exceptions_handlers = None
    validations = [('response_unique_values_ratio', operator.gt, 0.1)]


class N100EstimatorsGradientBoostingRegression(
        TreeEstimatorAction[XGBRegressor]):
    """Gradient Boosting with 100 estimators regression action.
    Fits a XGBRegressor with parameter n_estimators=100
    https://xgboost.readthedocs.io/en/stable/python/examples/basic_walkthrough.html
    """

    action_name = 'Gradient Boosting Regression 100 Estimators'
    action_key = 'n_100_estimators_gradient_boosting_regression'
    estimator = XGBRegressor(n_estimators=100)
    params_grid = {}
    exceptions_handlers = None
    validations = [('response_unique_values_ratio', operator.gt, 0.1)]


class RandomForestRegression(TreeEstimatorAction[RandomForestRegressor]):
    """Random Forest regression action.
    Fits a Scikit-Learn RandomForestRegressor
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    """

    action_name = 'Random Forest Regression'
    action_key = 'random_forest_regression'
    estimator = RandomForestRegressor()
    params_grid = {'n_estimators': [10, 50],
                   'max_features': ['sqrt', 'log2']}
    exceptions_handlers = None
    validations = [('response_unique_values_ratio', operator.gt, 0.1)]


class N100EstimatorsRandomForestRegression(TreeEstimatorAction[RandomForestRegressor]):
    """Random Forest regression with 100 trees in the forest action.
    Fits a Scikit-Learn RandomForestRegressor with parameter n_estimators=100
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    """

    action_name = 'Random Forest Regression 100 Estimators'
    action_key = 'n_100_estimators_random_forest_regression'
    estimator = RandomForestRegressor(n_estimators=100)
    params_grid = {'max_features': ['sqrt', 'log2']}
    exceptions_handlers = None
    validations = [('response_unique_values_ratio', operator.gt, 0.1)]
