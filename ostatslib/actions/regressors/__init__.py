"""Regression actions module
"""

from .decision_trees import (DecisionTreeRegression,)

from .ensembles import (RandomForestRegression,
                        N100EstimatorsRandomForestRegression,
                        AdaBoostRegression,
                        AdaBoostExponentialLossRegression,
                        AdaBoostSquareLossRegression,
                        BaggingRegression,
                        ExtraTreesRegression,
                        GradientBoostingRegression,
                        N100EstimatorsGradientBoostingRegression)

from .linear_models import (OLSLinearRegression,
                            PoissonRegression,
                            GammaRegression,
                            AutoARIMARegression)

from .svr import (LinearSupportVectorRegression,
                  PolyKernelSupportVectorRegression,
                  SupportVectorRegression,
                  NuPolyKernelSupportVectorRegression,
                  NuSupportVectorRegression,
                  NuLinearKernelSupportVectorRegression)
