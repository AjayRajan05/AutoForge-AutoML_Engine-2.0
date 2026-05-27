"""
🚀 AutoForge - Intelligent AutoML System
A unified, intelligent, and reliable automated machine learning platform
"""

from .core.unified_automl import UnifiedAutoML
from .core.bulletproof_error_handler import BulletproofErrorHandler
from .core.estimator import AutoForgeClassifier, AutoForgeRegressor, AutoForgeEstimator

__version__ = "1.0.0"
__author__ = "AutoForge Team"
__description__ = "Intelligent AutoML System with comprehensive capabilities"

__all__ = [
    'UnifiedAutoML',
    'BulletproofErrorHandler',
    'AutoForgeClassifier',
    'AutoForgeRegressor',
    'AutoForgeEstimator',
]


def create_automl(config=None):
    """Create AutoForge AutoML instance with optional configuration"""
    return UnifiedAutoML(config)


def quick_test():
    """Run a quick test of AutoForge"""
    print("Running AutoForge Quick Test...")

    try:
        import pandas as pd
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=200, n_features=8, n_informative=4, random_state=42
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="target")

        model = AutoForgeClassifier(model_family="ml", max_trials=5)
        model.fit(X, y)
        stats = model.automl_.get_performance_stats()
        print("Quick Test Complete!")
        print(f"Best Score: {stats.get('best_score', 'N/A')}")
        print(f"Task Type: {stats.get('task_type', 'N/A')}")
        print(f"Model Type: {stats.get('model_type', 'N/A')}")
        return model

    except Exception as e:
        print(f"Quick Test Failed: {e}")
        return None
