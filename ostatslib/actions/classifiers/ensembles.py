import operator
from sklearn.ensemble import (AdaBoostClassifier,
                              BaggingClassifier,
                              ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)

from ostatslib.actions.base import TargetModelEstimatorAction, TreeEstimatorAction


class AdaBoostClassification(TargetModelEstimatorAction[AdaBoostClassifier]):

    action_name = 'AdaBoost'
    action_key = 'adaboost'
    estimator = AdaBoostClassifier()
    params_grid = {'n_estimators': [10, 50, 100]}
    exceptions_handlers = None
    validations = [('is_response_discrete', operator.gt, 0),
                   ('response_unique_values_ratio', operator.ne, 0),
                   ('response_unique_values_ratio', operator.lt, 0.1)]


class BaggingClassification(TargetModelEstimatorAction[BaggingClassifier]):

    action_name = 'Bagging'
    action_key = 'bagging'
    estimator = BaggingClassifier()
    params_grid = {'n_estimators': [5, 10, 25]}
    exceptions_handlers = None
    validations = [('is_response_discrete', operator.gt, 0),
                   ('response_unique_values_ratio', operator.ne, 0),
                   ('response_unique_values_ratio', operator.lt, 0.1)]


class ExtraTreesClassification(TreeEstimatorAction[ExtraTreesClassifier]):

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


class GradientBoostingClassification(TargetModelEstimatorAction[GradientBoostingClassifier]):

    action_name = 'Gradient Boosting'
    action_key = 'gradient_boosting'
    estimator = GradientBoostingClassifier()
    params_grid = {'n_estimators': [10, 50, 100],
                   'loss': ['log_loss', 'deviance', 'exponential'],
                   'criterion': ['friedman_mse', 'squared_error'],
                   'max_features': ['sqrt', 'log2']}
    exceptions_handlers = None
    validations = [('is_response_discrete', operator.gt, 0),
                   ('response_unique_values_ratio', operator.ne, 0),
                   ('response_unique_values_ratio', operator.lt, 0.1)]


class RandomForestClassification(TreeEstimatorAction[RandomForestClassifier]):

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
