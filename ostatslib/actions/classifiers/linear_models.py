"""Linear models classification actions module
"""

import operator
from sklearn.linear_model import LogisticRegression

from ostatslib.actions.base import TargetModelEstimatorAction


class LogisticRegressionClassification(TargetModelEstimatorAction[LogisticRegression]):
    """Logistic Regression classification action.
    Fits a Scikit-Learn LogisticRegression with L2 regularization
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """

    action_name = 'Logistic Regression'
    action_key = 'logistic_regression'
    estimator = LogisticRegression()
    params_grid = {'C': [0.1, 1, 10]}
    exceptions_handlers = None
    validations = [('is_response_discrete', operator.gt, 0),
                   ('response_unique_values_ratio', operator.ne, 0),
                   ('response_unique_values_ratio', operator.lt, 0.1)]


class L1LogisticRegressionClassification(TargetModelEstimatorAction[LogisticRegression]):
    """Logistic Regression classification action.
    Fits a Scikit-Learn LogisticRegression with L1 regularization
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """

    action_name = 'Logistic Regression with L1 Regularization'
    action_key = 'l1_logistic_regression'
    estimator = LogisticRegression(penalty='l1', solver='liblinear')
    params_grid = {'C': [0.1, 1, 10]}
    exceptions_handlers = None
    validations = [('is_response_discrete', operator.gt, 0),
                   ('response_unique_values_ratio', operator.ne, 0),
                   ('response_unique_values_ratio', operator.lt, 0.1)]


class ElasticNetLogisticRegressionClassification(TargetModelEstimatorAction[LogisticRegression]):
    """Logistic Regression classification action.
    Fits a Scikit-Learn LogisticRegression with Elastic-Net regularization
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """

    action_name = 'Elastic-Net Logistic Regression'
    action_key = 'elasticnet_logistic_regression'
    estimator = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)
    exceptions_handlers = None
    validations = [('is_response_discrete', operator.gt, 0),
                   ('response_unique_values_ratio', operator.ne, 0),
                   ('response_unique_values_ratio', operator.lt, 0.1),
                   ('log_rows_count', operator.gt, 0),
                   ('log_rows_count', operator.lt, 81)]
