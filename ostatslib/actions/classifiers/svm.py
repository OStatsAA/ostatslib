"""Support Vector Machines classification actions module
"""

import operator
from sklearn.svm import LinearSVC, SVC, NuSVC
from ostatslib.actions.base import TargetModelEstimatorAction

_SVM_VALIDATIONS = [('is_response_discrete', operator.gt, 0),
                    ('log_rows_count', operator.gt, 0),
                    ('response_unique_values_ratio', operator.ne, 0),
                    ('response_unique_values_ratio', operator.lt, 0.1)]


class LinearSupportVectorClassification(TargetModelEstimatorAction[LinearSVC]):
    """Linear support vector classification action.
    Fits a Scikit-Learn LinearSVC
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    """

    action_name = 'Linear Support Vector Classification'
    action_key = 'linear_support_vector_classification'
    estimator = LinearSVC(dual=True)
    params_grid = {'loss': ['squared_hinge', 'hinge'],
                   'C': [0.1, 1, 10]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.81),
                   *_SVM_VALIDATIONS]


class NuSupportVectorClassification(TargetModelEstimatorAction[NuSVC]):
    """Nu support vector classification action.
    Fits a Scikit-Learn NuSVC
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html
    """

    action_name = 'Nu-Support Vector Classification'
    action_key = 'nu_support_vector_classification'
    estimator = NuSVC()
    params_grid = {'kernel': ['rbf']}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]

class NuLinearKernelSupportVectorClassification(TargetModelEstimatorAction[NuSVC]):
    """Nu support vector classification action with linear kernel.
    Fits a Scikit-Learn NuSVC
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html
    """

    action_name = 'Nu-Linear Kernel Support Vector Classification'
    action_key = 'nu_linear_kernel_support_vector_classification'
    estimator = NuSVC()
    params_grid = {'kernel': ['linear']}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]


class NuPolyKernelSupportVectorClassification(TargetModelEstimatorAction[NuSVC]):
    """Nu polynomial support vector classification action.
    Fits a Scikit-Learn NuSVC with polynomial kernel of degree 3 or 4
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html
    """

    action_name = 'Nu-Support Vector Classification with Polynomial Kernel'
    action_key = 'nu_poly_kernel_support_vector_classification'
    estimator = NuSVC(kernel='poly')
    params_grid = {'degree': [3, 4]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]


class SupportVectorClassification(TargetModelEstimatorAction[SVC]):
    """Support vector classification action.
    Fits a Scikit-Learn SVC
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """

    action_name = 'Support Vector Classification'
    action_key = 'support_vector_classification'
    estimator = SVC()
    params_grid = {'C': [0.1, 1, 10],
                   'kernel': ['linear', 'rbf']}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]


class PolyKernelSupportVectorClassification(TargetModelEstimatorAction[SVC]):
    """Polynomial support vector classification action.
    Fits a Scikit-Learn SVC with polynomial kernel of degree 3 or 4
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """

    action_name = 'Support Vector Classification with Polynomial Kernel'
    action_key = 'poly_kernel_support_vector_classification'
    estimator = SVC(kernel='poly')
    params_grid = {'C': [0.1, 1, 10],
                   'degree': [3, 4]}
    exceptions_handlers = None
    validations = [('log_rows_count', operator.lt, 0.71),
                   *_SVM_VALIDATIONS]
