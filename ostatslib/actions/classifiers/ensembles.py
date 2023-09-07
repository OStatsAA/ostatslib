"""Ensembles classification actions module
"""

import operator
from sklearn.ensemble import (AdaBoostClassifier,
                              BaggingClassifier,
                              ExtraTreesClassifier,
                              RandomForestClassifier)
from xgboost import XGBClassifier

from ostatslib.actions.base import TargetModelEstimatorAction, TreeEstimatorAction


class AdaBoostClassification(TargetModelEstimatorAction[AdaBoostClassifier]):
    """AdaBoost classification action.
    Fits a Scikit-Learn AdaBoostClassifier
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    """

    action_name = 'AdaBoost'
    action_key = 'adaboost'
    estimator = AdaBoostClassifier()
    params_grid = {'n_estimators': [10, 50, 100]}
    exceptions_handlers = None
    validations = [('is_response_discrete', operator.gt, 0),
                   ('response_unique_values_ratio', operator.ne, 0),
                   ('response_unique_values_ratio', operator.lt, 0.1)]


class BaggingClassification(TargetModelEstimatorAction[BaggingClassifier]):
    """Bagging classification action.
    Fits a Scikit-Learn BaggingClassifier
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
    """

    action_name = 'Bagging'
    action_key = 'bagging'
    estimator = BaggingClassifier()
    params_grid = {'n_estimators': [5, 10, 25]}
    exceptions_handlers = None
    validations = [('is_response_discrete', operator.gt, 0),
                   ('response_unique_values_ratio', operator.ne, 0),
                   ('response_unique_values_ratio', operator.lt, 0.1)]


class ExtraTreesClassification(TreeEstimatorAction[ExtraTreesClassifier]):
    """ExtraTrees classification action.
    Fits a Scikit-Learn ExtraTreesClassifier
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    """

    action_name = 'Extra-Trees'
    action_key = 'extra_trees'
    estimator = ExtraTreesClassifier()
    params_grid = {'n_estimators': [10, 50, 100],
                   'criterion': ['gini', 'log_loss'],
                   'max_features': ['sqrt', 'log2'],
                   'ccp_alpha': [1e-2, 1e-3]}
    exceptions_handlers = None
    validations = [('is_response_discrete', operator.gt, 0),
                   ('response_unique_values_ratio', operator.ne, 0),
                   ('response_unique_values_ratio', operator.lt, 0.1)]


class GradientBoostingClassification(TreeEstimatorAction[XGBClassifier]):
    """Gradient Boosting classification action.
    Fits a XGBoostClassfier
    https://xgboost.readthedocs.io/en/stable/python/examples/basic_walkthrough.html
    """

    action_name = 'Gradient Boosting'
    action_key = 'gradient_boosting'
    estimator = XGBClassifier()
    params_grid = {'n_estimators': [10, 50]}
    exceptions_handlers = None
    validations = [('is_response_discrete', operator.gt, 0),
                   ('response_unique_values_ratio', operator.ne, 0),
                   ('response_unique_values_ratio', operator.lt, 0.1)]


class N100GradientBoostingClassification(TreeEstimatorAction[XGBClassifier]):
    """Gradient Boosting with 100 estimators classification action.
    Fits a XGBoostClassfier with parameter n_estimators=100
    https://xgboost.readthedocs.io/en/stable/python/examples/basic_walkthrough.html
    """

    action_name = 'Gradient Boosting 100 Estimators'
    action_key = 'n_100_gradient_boosting'
    estimator = XGBClassifier(n_estimators=100)
    params_grid = {}
    exceptions_handlers = None
    validations = [('is_response_discrete', operator.gt, 0),
                   ('response_unique_values_ratio', operator.ne, 0),
                   ('response_unique_values_ratio', operator.lt, 0.1)]


class RandomForestClassification(TreeEstimatorAction[RandomForestClassifier]):
    """Random Forest classification action.
    Fits a Scikit-Learn RandomForestClassifier
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    action_name = 'Random Forest'
    action_key = 'random_forest'
    estimator = RandomForestClassifier()
    params_grid = {'n_estimators': [10, 20, 50],
                   'criterion': ['gini', 'log_loss'],
                   'max_features': ['sqrt', 'log2'],
                   'ccp_alpha': [1e-2, 1e-3]}
    exceptions_handlers = None
    validations = [('is_response_discrete', operator.gt, 0),
                   ('response_unique_values_ratio', operator.ne, 0),
                   ('response_unique_values_ratio', operator.lt, 0.1)]


class N100RandomForestClassification(TreeEstimatorAction[RandomForestClassifier]):
    """Random Forest classification with 100 trees in the forest action.
    Fits a Scikit-Learn RandomForestClassifier with parameter n_estimators=100
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    action_name = 'Random Forest 100 Estimators'
    action_key = 'n_100_estimators_random_forest'
    estimator = RandomForestClassifier(n_estimators=100)
    params_grid = {'criterion': ['gini', 'log_loss'],
                   'max_features': ['sqrt', 'log2'],
                   'ccp_alpha': [1e-2, 1e-3]}
    exceptions_handlers = None
    validations = [('is_response_discrete', operator.gt, 0),
                   ('response_unique_values_ratio', operator.ne, 0),
                   ('response_unique_values_ratio', operator.lt, 0.1)]
