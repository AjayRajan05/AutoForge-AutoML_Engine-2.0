"""
AutoForge API — integration layer and deprecated legacy entry points.
"""

from api.api_integration import APIIntegrator, api_integrator, integrate_with_autoforge

try:
    from core.unified_automl import UnifiedAutoML
except ImportError:
    UnifiedAutoML = None

try:
    from core.estimator import AutoForgeClassifier, AutoForgeRegressor, AutoForgeEstimator
except ImportError:
    AutoForgeClassifier = None
    AutoForgeRegressor = None
    AutoForgeEstimator = None

try:
    from api.automl import AutoML
except ImportError:
    AutoML = None

__all__ = [
    "APIIntegrator",
    "api_integrator",
    "integrate_with_autoforge",
    "UnifiedAutoML",
    "AutoForgeClassifier",
    "AutoForgeRegressor",
    "AutoForgeEstimator",
    "AutoML",
]
