"""Classification actions module
"""

from .decision_trees import (DecisionTreeClassification,
                             DecisionTreeEntropyCriteriaClassification,
                             DecisionTreeLogLossCriteriaClassification)

from .ensembles import (RandomForestClassification,
                        N100RandomForestClassification,
                        AdaBoostClassification,
                        BaggingClassification,
                        ExtraTreesClassification,
                        GradientBoostingClassification,
                        N100GradientBoostingClassification)

from .linear_models import (LogisticRegressionClassification,
                            L1LogisticRegressionClassification,
                            ElasticNetLogisticRegressionClassification,)

from .svm import (LinearSupportVectorClassification,
                  PolyKernelSupportVectorClassification,
                  SupportVectorClassification,
                  NuPolyKernelSupportVectorClassification,
                  NuSupportVectorClassification,
                  NuLinearKernelSupportVectorClassification)
