"""Decision Trees classification actions module
"""

import operator
from sklearn.tree import DecisionTreeClassifier

from ostatslib.actions.base import TreeEstimatorAction


class DecisionTreeClassification(TreeEstimatorAction[DecisionTreeClassifier]):
    """DecisionTree classification action.
    Fits a Scikit-Learn DecisionTreeClassifier
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """

    action_name = 'Decision Tree'
    action_key = 'decision_tree'
    estimator = DecisionTreeClassifier(criterion='gini')
    params_grid = {'max_features': ['sqrt', 'log2'],
                   'ccp_alpha': [1e-2, 1e-3]}
    exceptions_handlers = None
    validations = [('is_response_discrete', operator.gt, 0)]


class DecisionTreeEntropyCriteriaClassification(TreeEstimatorAction[DecisionTreeClassifier]):
    """DecisionTree classification action using entropy as splitting criteria.
    Fits a Scikit-Learn DecisionTreeClassifier
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """

    action_name = 'Decision Tree with Entropy Criteria'
    action_key = 'entropy_criteria_decision_tree'
    estimator = DecisionTreeClassifier(criterion='entropy')
    params_grid = {'max_features': ['sqrt', 'log2'],
                   'ccp_alpha': [1e-2, 1e-3]}
    exceptions_handlers = None
    validations = [('is_response_discrete', operator.gt, 0)]


class DecisionTreeLogLossCriteriaClassification(TreeEstimatorAction[DecisionTreeClassifier]):
    """DecisionTree classification action using log loss as splitting criteria.
    Fits a Scikit-Learn DecisionTreeClassifier
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """

    action_name = 'Decision Tree with Log Loss Criteria'
    action_key = 'log_loss_criteria_decision_tree'
    estimator = DecisionTreeClassifier(criterion='log_loss')
    params_grid = {'max_features': ['sqrt', 'log2'],
                   'ccp_alpha': [1e-2, 1e-3]}
    exceptions_handlers = None
    validations = [('is_response_discrete', operator.gt, 0)]
