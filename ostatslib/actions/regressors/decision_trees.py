import operator
from sklearn.tree import DecisionTreeRegressor

from ostatslib.actions.base import TreeEstimatorAction


class DecisionTreeRegression(TreeEstimatorAction[DecisionTreeRegressor]):

    action_name = 'Decision Tree Regression'
    action_key = 'decision_tree_regression'
    estimator = DecisionTreeRegressor()
    params_grid = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                   'max_features': ['sqrt', 'log2'],
                   'ccp_alpha': [1e-2, 1e-3]}
    exceptions_handlers = None
    validations = [('response_unique_values_ratio', operator.gt, 0.1)]
