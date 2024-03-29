"""Support Vector Regression actions module
"""

import operator
from sklearn.svm import SVR, LinearSVR, NuSVR
from ostatslib.actions.base import TargetModelEstimatorAction

_SVM_VALIDATIONS = [('is_response_quantitative', operator.gt, 0),
                    ('log_rows_count', operator.gt, 0),
                    ('response_unique_values_ratio', operator.gt, 0.1)]


class LinearSupportVectorRegression(TargetModelEstimatorAction[LinearSVR]):
    """Linear support vector regression action.
    Fits a Scikit-Learn LinearSVR
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
    """

    action_name = 'Linear Support Vector Regression'
    action_key = 'linear_support_vector_regression'
    estimator = LinearSVR(dual=True)
    params_grid = {'penalty': ['l1', 'l2'],
                   'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                   'C': [0.1, 1, 10, 100]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.81),
                   *_SVM_VALIDATIONS]


class NuSupportVectorRegression(TargetModelEstimatorAction[NuSVR]):
    """Nu support vector regression action.
    Fits a Scikit-Learn NuSVR
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html
    """

    action_name = 'Nu-Support Vector Regression'
    action_key = 'nu_support_vector_regression'
    estimator = NuSVR(kernel='rbf')
    params_grid = {'nu': [0.33, 0.5, 0.66],
                   'C': [0.1, 1, 10]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]


class NuLinearKernelSupportVectorRegression(TargetModelEstimatorAction[NuSVR]):
    """Nu support vector regression action.
    Fits a Scikit-Learn NuSVR
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html
    """

    action_name = 'Nu-Linear Kernel Support Vector Regression'
    action_key = 'nu_linear_kernel_support_vector_regression'
    estimator = NuSVR(kernel='linear')
    params_grid = {'nu': [0.33, 0.5, 0.66],
                   'C': [0.1, 1, 10]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]


class NuPolyKernelSupportVectorRegression(TargetModelEstimatorAction[NuSVR]):
    """Nu polynomial support vector regression action.
    Fits a Scikit-Learn NuSVR with polynomial kernel of degree 3 or 4
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html
    """

    action_name = 'Nu-Support Vector Regression with Polynomial Kernel'
    action_key = 'nu_poly_kernel_support_vector_regression'
    estimator = NuSVR(kernel='poly')
    params_grid = {'nu': [0.33, 0.5, 0.66],
                   'C': [0.1, 1, 10],
                   'degree': [3, 4]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]


class SupportVectorRegression(TargetModelEstimatorAction[SVR]):
    """Support vector regression action.
    Fits a Scikit-Learn SVR
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    """

    action_name = 'Support Vector Regression'
    action_key = 'support_vector_regression'
    estimator = SVR()
    params_grid = {'C': [0.1, 1, 10],
                   'kernel': ['linear', 'rbf']}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]


class PolyKernelSupportVectorRegression(TargetModelEstimatorAction[SVR]):
    """Polynomial support vector regression action.
    Fits a Scikit-Learn SVR with polynomial kernel of degree 3 or 4
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    """

    action_name = 'Support Vector Regression with Polynomial Kernel'
    action_key = 'poly_kernel_support_vector_regression'
    estimator = SVR(kernel='poly')
    params_grid = {'C': [0.1, 1, 10],
                   'degree': [3, 4]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]
