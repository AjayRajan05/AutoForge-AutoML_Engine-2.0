"""
Advanced Models Support
LightGBM and Simple Neural Networks integration
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

logger = logging.getLogger(__name__)


class SimpleNeuralNetwork(BaseEstimator, ClassifierMixin, RegressorMixin):
    """
    Wrapper for sklearn neural networks with automatic architecture tuning
    """
    
    def __init__(self,
                 task_type: str = "classification",
                 max_layers: int = 3,
                 max_neurons: int = 128,
                 activation: str = "relu",
                 solver: str = "adam",
                 alpha: float = 0.0001,
                 learning_rate_init: float = 0.001,
                 max_iter: int = 1000,
                 early_stopping: bool = True):
        """
        Initialize simple neural network
        
        Args:
            task_type: "classification" or "regression"
            max_layers: Maximum number of hidden layers
            max_neurons: Maximum neurons per layer
            activation: Activation function
            solver: Solver for optimization
            alpha: L2 regularization parameter
            learning_rate_init: Initial learning rate
            max_iter: Maximum iterations
            early_stopping: Whether to use early stopping
        """
        self.task_type = task_type
        self.max_layers = max_layers
        self.max_neurons = max_neurons
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _create_model(self, input_dim: int) -> Union[MLPClassifier, MLPRegressor]:
        """
        Create neural network model based on input dimensions
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Neural network model (classifier or regressor)
            
        Raises:
            ValueError: If input_dim is invalid
        """
        # Input validation
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        
        # Determine architecture based on input size
        
        # Determine architecture based on input size
        if input_dim < 10:
            hidden_layer_sizes = (max(10, input_dim * 2),)
        elif input_dim < 50:
            hidden_layer_sizes = (max(20, input_dim), max(10, input_dim // 2))
        elif input_dim < 100:
            hidden_layer_sizes = (max(50, input_dim // 2), max(25, input_dim // 4), max(10, input_dim // 8))
        else:
            hidden_layer_sizes = (max(64, input_dim // 4), max(32, input_dim // 8), max(16, input_dim // 16))
        
        # Limit neurons to max_neurons
        hidden_layer_sizes = tuple(min(n, self.max_neurons) for n in hidden_layer_sizes)
        
        # Limit layers to max_layers
        hidden_layer_sizes = hidden_layer_sizes[:self.max_layers]
        
        if self.task_type == "classification":
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                early_stopping=self.early_stopping,
                random_state=42,
                verbose=False
            )
        else:
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                early_stopping=self.early_stopping,
                random_state=42,
                verbose=False
            )
        
        return model
    
    def fit(self, X, y):
        """Fit the neural network model with enhanced parameter validation"""
        try:
            # Enhanced parameter validation for revolutionary dynamic suggestions
            if not hasattr(self, 'task_type') or self.task_type is None:
                self.task_type = "classification"  # Default
            
            # Validate and sanitize parameters dynamically
            if not hasattr(self, 'max_layers') or self.max_layers is None:
                self.max_layers = 2
            elif self.max_layers < 1:
                self.max_layers = 1
            elif self.max_layers > 10:  # Reasonable upper limit
                self.max_layers = 10
            
            if not hasattr(self, 'max_neurons') or self.max_neurons is None:
                self.max_neurons = 64
            elif self.max_neurons < 8:
                self.max_neurons = 8
            elif self.max_neurons > 1024:  # Reasonable upper limit
                self.max_neurons = 1024
                
            if not hasattr(self, 'activation') or self.activation is None:
                self.activation = "relu"
            elif self.activation not in ["relu", "tanh", "logistic"]:
                self.activation = "relu"  # Default to safe option
                
            if not hasattr(self, 'solver') or self.solver is None:
                self.solver = "adam"
            elif self.solver not in ["adam", "sgd"]:
                self.solver = "adam"  # Default to safe option
                
            if not hasattr(self, 'alpha') or self.alpha is None:
                self.alpha = 0.0001
            elif self.alpha < 1e-6:
                self.alpha = 1e-6
            elif self.alpha > 1.0:
                self.alpha = 1.0
                
            if not hasattr(self, 'learning_rate_init') or self.learning_rate_init is None:
                self.learning_rate_init = 0.001
            elif self.learning_rate_init < 1e-5:
                self.learning_rate_init = 1e-5
            elif self.learning_rate_init > 1.0:
                self.learning_rate_init = 1.0
                
            if not hasattr(self, 'max_iter') or self.max_iter is None:
                self.max_iter = 500
            elif self.max_iter < 100:
                self.max_iter = 100
            elif self.max_iter > 10000:  # Reasonable upper limit
                self.max_iter = 10000
                
            if not hasattr(self, 'early_stopping') or self.early_stopping is None:
                self.early_stopping = True
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Create model
            self.model = self._create_model(X_scaled.shape[1])
            
            # Fit model
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            
            logger.info(f"Neural Network fitted: {len(self.model.hidden_layer_sizes)} layers, "
                       f"architecture: {self.model.hidden_layer_sizes}")
            
        except Exception as e:
            logger.error(f"Failed to fit neural network: {e}")
            raise
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities (classification only)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for sklearn compatibility"""
        return {
            "task_type": self.task_type,
            "max_layers": self.max_layers,
            "max_neurons": self.max_neurons,
            "activation": self.activation,
            "solver": self.solver,
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "early_stopping": self.early_stopping
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def _more_tags(self):
        """For sklearn compatibility"""
        return {
            'requires_positive_X': False,
            'requires_fit': True,
        }


class LightGBMWrapper(BaseEstimator, ClassifierMixin, RegressorMixin):
    """
    Wrapper for LightGBM models with automatic parameter tuning
    """
    
    def __init__(self,
                 task_type: str = "classification",
                 n_estimators: int = 100,
                 max_depth: int = -1,
                 learning_rate: float = 0.1,
                 num_leaves: int = 31,
                 feature_fraction: float = 0.8,
                 bagging_fraction: float = 0.8,
                 bagging_freq: int = 5,
                 min_child_samples: int = 20,
                 reg_alpha: float = 0.0,
                 reg_lambda: float = 0.0,
                 random_state: int = 42):
        """
        Initialize LightGBM wrapper
        
        Args:
            task_type: "classification" or "regression"
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (-1 for no limit)
            learning_rate: Learning rate
            num_leaves: Maximum number of leaves in one tree
            feature_fraction: Fraction of features to use for each iteration
            bagging_fraction: Fraction of data to use for each iteration
            bagging_freq: Frequency for bagging
            min_child_samples: Minimum number of data needed in a child
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random seed
        """
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        
        self.model = None
        self.is_fitted = False
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
    
    def _create_model(self):
        """Create LightGBM model"""
        
        # Check if LightGBM is available
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Install with: pip install lightgbm")
        
        # Determine objective based on task type
        if self.task_type == "classification":
            objective = "binary" if len(np.unique([0, 1])) == 2 else "multiclass"
            model = lgb.LGBMClassifier(
                objective=objective,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                feature_fraction=self.feature_fraction,
                bagging_fraction=self.bagging_fraction,
                bagging_freq=self.bagging_freq,
                min_child_samples=self.min_child_samples,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                verbose=-1  # Suppress LightGBM output
            )
        else:
            model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                feature_fraction=self.feature_fraction,
                bagging_fraction=self.bagging_fraction,
                bagging_freq=self.bagging_freq,
                min_child_samples=self.min_child_samples,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                verbose=-1  # Suppress LightGBM output
            )
        
        return model
    
    def fit(self, X, y):
        """Fit the LightGBM model"""
        try:
            # Create model
            self.model = self._create_model()
            
            # Fit model
            self.model.fit(X, y)
            self.is_fitted = True
            
            logger.info(f"LightGBM fitted: {self.task_type}, {self.n_estimators} estimators")
            
        except Exception as e:
            logger.error(f"Failed to fit LightGBM: {e}")
            raise
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities (classification only)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")
        
        return self.model.predict_proba(X)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for sklearn compatibility"""
        return {
            "task_type": self.task_type,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "min_child_samples": self.min_child_samples,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def _more_tags(self):
        """For sklearn compatibility"""
        return {
            'requires_positive_X': False,
            'requires_fit': True,
        }
    
    def feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance"""
        if not self.is_fitted:
            return None
        
        return self.model.feature_importances_


class AdvancedModelFactory:
    """
    Factory for creating advanced models
    """
    
    @staticmethod
    def create_lightgbm(task_type: str = "classification", **kwargs) -> LightGBMWrapper:
        """Create LightGBM model"""
        return LightGBMWrapper(task_type=task_type, **kwargs)
    
    @staticmethod
    def create_neural_network(task_type: str = "classification", **kwargs) -> SimpleNeuralNetwork:
        """Create Neural Network model"""
        return SimpleNeuralNetwork(task_type=task_type, **kwargs)
    
    @staticmethod
    def get_available_models() -> Dict[str, List[str]]:
        """Get list of available advanced models"""
        models = {
            "neural_network": ["classification", "regression"],
            "lightgbm": ["classification", "regression"] if LIGHTGBM_AVAILABLE else []
        }
        
        # Filter out models that aren't available
        available_models = {}
        for model_name, task_types in models.items():
            if task_types:  # Only include if there are available task types
                available_models[model_name] = task_types
        
        return available_models
    
    @staticmethod
    def create_model(model_name: str, task_type: str = "classification", **kwargs):
        """Create model by name"""
        if model_name == "neural_network":
            return AdvancedModelFactory.create_neural_network(task_type, **kwargs)
        elif model_name == "lightgbm":
            return AdvancedModelFactory.create_lightgbm(task_type, **kwargs)
        else:
            raise ValueError(f"Unknown model: {model_name}")


# Convenience functions
def create_lightgbm_classifier(**kwargs) -> LightGBMWrapper:
    """Create LightGBM classifier"""
    return AdvancedModelFactory.create_lightgbm("classification", **kwargs)


def create_lightgbm_regressor(**kwargs) -> LightGBMWrapper:
    """Create LightGBM regressor"""
    return AdvancedModelFactory.create_lightgbm("regression", **kwargs)


def create_neural_network_classifier(**kwargs) -> SimpleNeuralNetwork:
    """Create Neural Network classifier"""
    return AdvancedModelFactory.create_neural_network("classification", **kwargs)


def create_neural_network_regressor(**kwargs) -> SimpleNeuralNetwork:
    """Create Neural Network regressor"""
    return AdvancedModelFactory.create_neural_network("regression", **kwargs)
