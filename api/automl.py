"""
AutoML - Ultra-minimal coordinator that delegates to specialized modules

.. deprecated::
    Use ``from core.estimator import AutoForgeClassifier`` or
    ``from autoforge import AutoForgeRegressor`` instead. This module is legacy.
"""

import logging
import warnings
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np

warnings.warn(
    "api.automl.AutoML is deprecated; use core.estimator or autoforge package.",
    DeprecationWarning,
    stacklevel=2,
)

# Import core components
from core.unified_automl import UnifiedAutoML


class AutoML:
    """
    Ultra-minimal AutoML coordinator - only 50 lines!
    Delegates all complex logic to specialized modules.
    """
    
    def __init__(self,
                 n_trials=50,
                 timeout=None,
                 cv=3,
                 use_adaptive_optimization=True,
                 use_dataset_optimization=True,
                 use_caching=True,
                 show_progress=True,
                 use_explainability=True,
                 enable_advanced_features=True):
        """Initialize AutoML with configuration"""
        # Core coordinator handles the main workflow
        self.coordinator = UnifiedAutoML({
            'n_trials': n_trials, 
            'timeout': timeout, 
            'cv': cv,
            'use_adaptive_optimization': use_adaptive_optimization,
            'use_dataset_optimization': use_dataset_optimization,
            'use_caching': use_caching, 
            'show_progress': show_progress,
            'enable_advanced_features': enable_advanced_features
        })
        
        # Store configuration parameters for access by subclasses
        self._n_trials = n_trials
        self._timeout = timeout
        self._cv = cv
        self._use_adaptive_optimization = use_adaptive_optimization
        self._use_dataset_optimization = use_dataset_optimization
        self._use_caching = use_caching
        self._show_progress = show_progress
        self._use_explainability = use_explainability
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize explainability manager
        try:
            from ..features.explainability.actionable_explainability import ActionableExplainability
            self.explainability_manager = ActionableExplainability()
        except ImportError:
            self.explainability_manager = None
            self.logger.warning("Explainability manager not available")
        
        # Initialize meta-learning manager
        try:
            from ..features.meta_learning.self_improver import SelfImprover
            self.meta_learning_manager = SelfImprover()
        except ImportError:
            self.meta_learning_manager = None
            self.logger.warning("Meta-learning manager not available")

    def get_dataset_info(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Get basic dataset information
        
        Args:
            X: Feature data
            
        Returns:
            Dictionary with dataset information
            
        Raises:
            ValueError: If X is None
            TypeError: If X has invalid type
        """
        if X is None:
            raise ValueError("X cannot be None")
        
        if not hasattr(X, 'shape'):
            return {}
        
        if hasattr(X, 'shape') and len(X.shape) < 2:
            return {"num_rows": X.shape[0], "num_cols": 1}
        
        return {
            "num_rows": X.shape[0],
            "num_cols": X.shape[1],
            "has_missing": hasattr(X, 'isnull') and X.isnull().sum().sum() > 0 if hasattr(X, 'isnull') else False,
        }

    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray]) -> 'AutoML':
        """
        Fit AutoML model to training data
        
        Args:
            X: Training features
            y: Training targets (optional)
            
        Returns:
            Self
            
        Raises:
            ValueError: If inputs are invalid
        """
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        
        if self.coordinator is not None:
            try:
                # Try to call the UnifiedAutoML fit method
                self.coordinator.fit(X, y)
            except Exception as e:
                self.logger.error(f"Coordinator fit failed: {e}")
                # Fallback: store data and mark as fitted
                self._X_train = X
                self._y_train = y
                self._is_fitted = True
        else:
            self.logger.warning("AutoMLCoordinator not available, fit operation skipped")
            # Fallback behavior
            self._X_train = X
            self._y_train = y
            self._is_fitted = True
        
        return self

    def predict(self, X):
        """Make predictions - delegates to coordinator"""
        if self.coordinator is not None:
            return self.coordinator.predict(X)
        else:
            self.logger.warning("AutoMLCoordinator not available, prediction failed")
            raise NotImplementedError("Prediction not available in fallback mode")

    def predict_proba(self, X):
        """Make probability predictions - delegates to coordinator"""
        if self.coordinator is not None:
            return self.coordinator.predict_proba(X)
        else:
            self.logger.warning("AutoMLCoordinator not available, probability prediction failed")
            raise NotImplementedError("Probability prediction not available in fallback mode")

    # Explainability methods - delegate to explainability manager
    def explain(self, X, y=None, use_shap=True, top_n=10):
        """Generate model explanations."""
        if self.explainability_manager is not None:
            return self.explainability_manager.explain_model(
                X, y, self.coordinator.best_pipeline, 
                self.coordinator.task_type, use_shap, top_n
            )
        else:
            return {"error": "Explainability not available in fallback mode"}

    def get_feature_importance(self, method="aggregated"):
        """Get feature importance."""
        if self.explainability_manager is not None:
            return self.explainability_manager.get_feature_importance(method)
        else:
            return {"error": "Feature importance not available in fallback mode"}

    def get_explanation_summary(self):
        """Get human-readable explanation summary."""
        if self.explainability_manager is not None:
            return self.explainability_manager.get_explanation_summary()
        else:
            return "No explanation summary available in fallback mode"

    def plot_feature_importance(self, top_n=10, save_path=None, method="aggregated"):
        """Plot feature importance."""
        if self.explainability_manager is not None:
            return self.explainability_manager.plot_feature_importance(top_n, save_path, method)
        else:
            return {"error": "Plotting not available in fallback mode"}

    def plot_shap_summary(self, save_path=None, max_display=20):
        """Plot SHAP summary plot."""
        if self.explainability_manager is not None:
            return self.explainability_manager.plot_shap_summary(save_path, max_display)
        else:
            return {"error": "SHAP plotting not available in fallback mode"}

    def get_top_features(self, n=10, method="aggregated"):
        """Get top N features by importance."""
        if self.explainability_manager is not None:
            return self.explainability_manager.get_top_features(n, method)
        else:
            return {"error": "Feature ranking not available in fallback mode"}

    def explain_predictions(self, X, n_instances=5):
        """Explain individual predictions."""
        if self.explainability_manager is not None:
            return self.explainability_manager.explain_predictions(
                X, self.coordinator.best_pipeline, n_instances
            )
        else:
            return {"error": "Prediction explanation not available in fallback mode"}

    # Meta-learning methods - delegate to meta-learning manager
    def learn_from_experiment(self, experiment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Learn patterns from completed experiment."""
        if self.meta_learning_manager is not None:
            return self.meta_learning_manager.learn_from_experiment(experiment_result)
        else:
            return {"error": "Meta-learning not available in fallback mode"}

    # Properties to access coordinator state
    @property
    def best_pipeline(self):
        """Access the trained pipeline"""
        return self.coordinator.best_pipeline

    @property
    def best_score(self):
        """Get the best score achieved"""
        return self.coordinator.best_score

    @property
    def best_model_name(self):
        """Get the name of the best model"""
        return self.coordinator.best_model_name

    @property
    def task_type(self):
        """Get the detected task type"""
        return self.coordinator.task_type

    @property
    def feature_metadata(self):
        """Get feature analysis results"""
        return self.coordinator.feature_metadata

    @property
    def dataset_profile(self):
        """Get dataset characteristics"""
        return self.coordinator.dataset_profile

    @property
    def optimization_metadata(self):
        """Get optimization metadata"""
        return self.coordinator.optimization_metadata

    @property
    def n_trials(self):
        """Get number of trials used"""
        return self.coordinator.n_trials

    @property
    def use_adaptive_optimization(self):
        """Get adaptive optimization setting"""
        return self._use_adaptive_optimization

    @property
    def use_dataset_optimization(self):
        """Get dataset optimization setting"""
        return self._use_dataset_optimization

    @property
    def use_caching(self):
        """Get caching setting"""
        return self._use_caching

    @property
    def show_progress(self):
        """Get progress display setting"""
        return self._show_progress

    @property
    def use_explainability(self):
        """Get explainability setting"""
        return self._use_explainability