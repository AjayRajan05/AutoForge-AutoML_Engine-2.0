"""
AutoForge public package — canonical import namespace.

Prefer:
    from autoforge import AutoForgeRegressor, AutoForgeClassifier
"""

from core.estimator import (
    AutoForgeClassifier,
    AutoForgeEstimator,
    AutoForgeRegressor,
    autoforge_fit,
)
from core.unified_automl import UnifiedAutoML

__all__ = [
    "AutoForgeClassifier",
    "AutoForgeRegressor",
    "AutoForgeEstimator",
    "UnifiedAutoML",
    "autoforge_fit",
]

__version__ = "1.0.0"
