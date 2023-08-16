import operator
from sklearn.svm import LinearSVC, SVC, NuSVC
from ostatslib.actions.base import TargetModelEstimatorAction

_SVM_VALIDATIONS = [('is_response_discrete', operator.gt, 0),
                    ('log_rows_count', operator.gt, 0),
                    ('response_unique_values_ratio', operator.ne, 0),
                    ('response_unique_values_ratio', operator.lt, 0.1)]


class LinearSupportVectorClassification(TargetModelEstimatorAction[LinearSVC]):

    action_name = 'Linear Support Vector Classification'
    action_key = 'linear_support_vector_classification'
    estimator = LinearSVC(dual=True)
    params_grid = {'loss': ['squared_hinge', 'hinge'],
                   'C': [0.1, 1, 10],
                   'tol': [1e-2, 1e-3, 1e-4]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.81),
                   *_SVM_VALIDATIONS]


class NuSupportVectorClassification(TargetModelEstimatorAction[NuSVC]):

    action_name = 'Nu-Support Vector Classification'
    action_key = 'nu_support_vector_classification'
    estimator = NuSVC()
    params_grid = {'nu': [0.33, 0.5, 0.66],
                   'kernel': ['linear', 'rbf'],
                   'gamma': ['scale', 'auto'],
                   'tol': [1e-2, 1e-3, 1e-4]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]


class NuPolyKernelSupportVectorClassification(TargetModelEstimatorAction[NuSVC]):

    action_name = 'Nu-Support Vector Classification with Polynomial Kernel'
    action_key = 'nu_poly_kernel_support_vector_classification'
    estimator = NuSVC(kernel='poly')
    params_grid = {'nu': [0.33, 0.5, 0.66],
                   'degree': [3, 4],
                   'gamma': ['scale', 'auto'],
                   'tol': [1e-2, 1e-3, 1e-4]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]


class SupportVectorClassification(TargetModelEstimatorAction[SVC]):

    action_name = 'Support Vector Classification'
    action_key = 'support_vector_classification'
    estimator = SVC()
    params_grid = {'C': [0.1, 1, 10],
                   'kernel': ['linear', 'rbf'],
                   'gamma': ['scale', 'auto'],
                   'tol': [1e-2, 1e-3, 1e-4]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]


class PolyKernelSupportVectorClassification(TargetModelEstimatorAction[SVC]):

    action_name = 'Support Vector Classification with Polynomial Kernel'
    action_key = 'poly_kernel_support_vector_classification'
    estimator = SVC(kernel='poly')
    params_grid = {'C': [0.1, 1, 10],
                   'degree': [3, 4],
                   'gamma': ['scale', 'auto'],
                   'tol': [1e-2, 1e-3, 1e-4]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]
