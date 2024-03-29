"""
ModelsFeaturesSet module
"""

from dataclasses import dataclass, field
from gymnasium.spaces import Box

from ostatslib.states.features_set import FeaturesSet


@dataclass(init=False)
class ModelsFeaturesSet(FeaturesSet):
    """
    Class to hold features extracted from models fitting attempts.
    """
    are_linear_model_regression_residuals_correlated: int = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    are_linear_model_regression_residuals_homoscedastic: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    are_linear_model_regression_residuals_normally_distributed: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    is_linear_model_regression_recursive_residuals_mean_zero: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    does_poisson_regression_raises_perfect_separation_error: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    # classifiers scores

    # decision trees

    decision_tree_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    entropy_criteria_decision_tree_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    log_loss_criteria_decision_tree_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    # ensembles

    adaboost_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    bagging_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    extra_trees_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    gradient_boosting_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    n_100_gradient_boosting_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    random_forest_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    n_100_estimators_random_forest_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    # linear models

    logistic_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    l1_logistic_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    elasticnet_logistic_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    # support vector machines

    linear_support_vector_classification_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    nu_support_vector_classification_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    nu_linear_kernel_support_vector_classification_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    nu_poly_kernel_support_vector_classification_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    support_vector_classification_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    poly_kernel_support_vector_classification_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    linear_support_vector_classification_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    # regressors scores

    # decision trees

    decision_tree_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    # ensembles

    adaboost_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    adaboost_square_loss_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    adaboost_exponential_loss_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    bagging_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    extra_trees_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    gradient_boosting_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    n_100_estimators_gradient_boosting_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    random_forest_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    n_100_estimators_random_forest_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    # linear models

    poisson_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    gamma_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    linear_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    # support vector machines

    linear_support_vector_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    nu_support_vector_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    nu_linear_kernel_support_vector_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    nu_poly_kernel_support_vector_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    support_vector_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    poly_kernel_support_vector_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    # time series

    auto_arima_regression_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    # clustering scores

    kmeans_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    dbscan_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    spectral_clustering_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    spectral_clustering_discretize_labels_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })

    spectral_clustering_qr_labels_score_reward: float = field(
        default=0,
        metadata={
            'gym_space': Box(-1, 1),
            'get_value_fn': None
        })
