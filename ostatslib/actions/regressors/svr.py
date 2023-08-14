import operator
from sklearn.svm import SVR, LinearSVR, NuSVR
from ostatslib.actions.base import TargetModelEstimatorAction

_SVM_VALIDATIONS = [('is_response_quantitative', operator.gt, 0),
                    ('log_rows_count', operator.gt, 0),
                    ('response_unique_values_ratio', operator.gt, 0.1)]


class LinearSupportVectorRegression(TargetModelEstimatorAction[LinearSVR]):

    action_name = 'Linear Support Vector Regression'
    action_key = 'linear_support_vector_regression'
    estimator = LinearSVR(dual=True)
    params_grid = {'penalty': ['l1', 'l2'],
                   'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                   'C': [0.1, 1, 10, 100],
                   'tol': [1e-2, 1e-3, 1e-4]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.81),
                   *_SVM_VALIDATIONS]


class NuSupportVectorRegression(TargetModelEstimatorAction[NuSVR]):

    action_name = 'Nu-Support Vector Regression'
    action_key = 'nu_support_vector_regression'
    estimator = NuSVR()
    params_grid = {'nu': [0.33, 0.5, 0.66],
                   'C': [0.1, 1, 10, 100],
                   'kernel': ['linear', 'rbf', 'sigmoid'],
                   'gamma': ['scale', 'auto'],
                   'tol': [1e-2, 1e-3, 1e-4]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]


class NuPolyKernelSupportVectorRegression(TargetModelEstimatorAction[NuSVR]):

    action_name = 'Nu-Support Vector Regression with Polynomial Kernel'
    action_key = 'nu_poly_kernel_support_vector_regression'
    estimator = NuSVR(kernel='poly')
    params_grid = {'nu': [0.33, 0.5, 0.66],
                   'C': [0.1, 1, 10, 100],
                   'degree': [3, 4],
                   'gamma': ['scale', 'auto'],
                   'tol': [1e-3, 1e-4, 1e-5]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]


class SupportVectorRegression(TargetModelEstimatorAction[SVR]):

    action_name = 'Support Vector Regression'
    action_key = 'support_vector_regression'
    estimator = SVR()
    params_grid = {'C': [0.1, 1, 10, 100],
                   'kernel': ['linear', 'rbf', 'sigmoid'],
                   'gamma': ['scale', 'auto'],
                   'tol': [1e-2, 1e-3, 1e-4]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]


class PolyKernelSupportVectorRegression(TargetModelEstimatorAction[SVR]):

    action_name = 'Support Vector Regression with Polynomial Kernel'
    action_key = 'poly_kernel_support_vector_regression'
    estimator = SVR(kernel='poly')
    params_grid = {'C': [0.1, 1, 10, 100],
                   'degree': [3, 4],
                   'gamma': ['scale', 'auto'],
                   'tol': [1e-2, 1e-3, 1e-4]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]
