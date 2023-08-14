import operator
from sklearn.tree import DecisionTreeClassifier

from ostatslib.actions.base import TreeEstimatorAction


class DecisionTreeClassification(TreeEstimatorAction[DecisionTreeClassifier]):

    action_name = 'Decision Tree'
    action_key = 'decision_tree'
    estimator = DecisionTreeClassifier()
    params_grid = {'criterion': ['gini', 'entropy', 'log_loss'],
                   'max_features': ['sqrt', 'log2'],
                   'ccp_alpha': [1e-2, 1e-3]}
    exceptions_handlers = None
    validations = [('is_response_discrete', operator.gt, 0)]
