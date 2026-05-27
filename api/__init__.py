"""
AutoForge API — public AutoML interfaces.
"""

try:
    from .automl import AutoML
except ImportError:
    AutoML = None

try:
    from .enhanced_automl import EnhancedAutoML
except ImportError:
    EnhancedAutoML = None

try:
    from .self_improving_automl import SelfImprovingAutoML
except ImportError:
    SelfImprovingAutoML = None

try:
    from ..core.unified_automl import UnifiedAutoML
except ImportError:
    UnifiedAutoML = None

try:
    from ..core.estimator import AutoForgeClassifier, AutoForgeRegressor, AutoForgeEstimator
except ImportError:
    AutoForgeClassifier = None
    AutoForgeRegressor = None
    AutoForgeEstimator = None

__all__ = [
    'AutoML',
    'EnhancedAutoML',
    'SelfImprovingAutoML',
    'UnifiedAutoML',
    'AutoForgeClassifier',
    'AutoForgeRegressor',
    'AutoForgeEstimator',
]

AdvancedAutoML = EnhancedAutoML

# Deprecated: import explicitly from api.truly_bulletproof_automl if needed.
