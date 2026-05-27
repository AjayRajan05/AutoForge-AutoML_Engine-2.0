"""
🔧 AutoForge Core Module
Core components and unified AutoML system
"""

from .unified_automl import UnifiedAutoML
from .bulletproof_error_handler import BulletproofErrorHandler, bulletproof_method
from .estimator import (
    AutoForgeClassifier,
    AutoForgeRegressor,
    AutoForgeEstimator,
    autoforge_fit,
)

__all__ = [
    'UnifiedAutoML',
    'BulletproofErrorHandler',
    'bulletproof_method',
    'AutoForgeClassifier',
    'AutoForgeRegressor',
    'AutoForgeEstimator',
    'autoforge_fit',
]
