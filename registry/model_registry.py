"""
Runtime model registry facade.

Model class definitions and HPO spaces live in models/registry.py.
"""

import logging
from typing import Dict, Any, List, Optional, Type
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# Import the existing model registry
try:
    from models.registry import (
        MODEL_REGISTRY, CLASSIFICATION_MODELS, REGRESSION_MODELS,
        CLASSICAL_CLASSIFICATION, CLASSICAL_REGRESSION,
        get_model_class, get_available_models, is_advanced_model,
        create_model_instance, filter_models_by_family, SKLEARN_DL_MODELS,
    )
    HAS_ROOT_REGISTRY = True
except ImportError:
    HAS_ROOT_REGISTRY = False
    MODEL_REGISTRY = {}
    CLASSIFICATION_MODELS = {}
    REGRESSION_MODELS = {}
    CLASSICAL_CLASSIFICATION = []
    CLASSICAL_REGRESSION = []
    SKLEARN_DL_MODELS = set()
    filter_models_by_family = lambda models, model_family='ml': models  # noqa: E731

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Centralized registry for machine learning models
    Uses existing root-level registry when available
    """
    
    def __init__(self):
        """Initialize model registry"""
        if HAS_ROOT_REGISTRY:
            self.models = MODEL_REGISTRY
            self.classification_models = CLASSIFICATION_MODELS
            self.regression_models = REGRESSION_MODELS
        else:
            # Fallback to minimal registry
            self.models = {}
            self.classification_models = {}
            self.regression_models = {}
        
        self.model_history = []
    
    def get_model(self, model_name: str, task_type: str = 'classification', **kwargs) -> BaseEstimator:
        """
        Get a model instance by name
        
        Args:
            model_name: Name of the model
            task_type: Type of task (classification/regression)
            **kwargs: Additional parameters for model initialization
            
        Returns:
            Model instance
        """
        if HAS_ROOT_REGISTRY:
            try:
                return create_model_instance(model_name, task_type=task_type, **kwargs)
            except Exception as e:
                logger.warning(f"Root registry failed for {model_name}: {e}")
        
        # Fallback logic
        if task_type == 'classification' and model_name in self.classification_models:
            model_class = self.classification_models[model_name]
            return model_class(**kwargs) if callable(model_class) else model_class(**kwargs)
        elif task_type == 'regression' and model_name in self.regression_models:
            model_class = self.regression_models[model_name]
            return model_class(**kwargs) if callable(model_class) else model_class(**kwargs)
        
        raise ValueError(f"Model '{model_name}' not found for task type '{task_type}'. Available models: {self.get_available_models(task_type)}")
    
    def get_available_models(self, task_type: str = None) -> Dict[str, List[str]]:
        """
        Get list of available models
        
        Args:
            task_type: Type of task (classification/regression). If None, returns all
            
        Returns:
            Dictionary of available models by category
        """
        if HAS_ROOT_REGISTRY:
            if task_type == 'classification':
                return {'default': list(CLASSIFICATION_MODELS.keys()), 'advanced': []}
            elif task_type == 'regression':
                return {'default': list(REGRESSION_MODELS.keys()), 'advanced': []}
            else:
                return {'default': list(MODEL_REGISTRY.keys()), 'advanced': []}
        
        # Fallback
        available = {'default': [], 'advanced': []}
        if task_type in ['classification', None]:
            available['default'].extend(self.classification_models.keys())
        if task_type in ['regression', None]:
            available['default'].extend(self.regression_models.keys())
        
        return available
    
    def recommend_models(
        self,
        n_samples: int,
        n_features: int,
        task_type: str,
        model_family: str = 'ml',
    ) -> List[str]:
        """Recommend models based on dataset size, task type, and model family."""
        if model_family == 'dl':
            try:
                from execution.dl_trainer import DL_AVAILABLE
                if DL_AVAILABLE:
                    return ['dnn']
            except ImportError:
                pass
            if task_type == 'regression':
                return ['neural_network_regressor']
            return ['neural_network']

        if task_type == 'regression':
            canon = CLASSICAL_REGRESSION if HAS_ROOT_REGISTRY else [
                'random_forest_regressor', 'ridge', 'lasso', 'elastic_net', 'xgboost_regressor'
            ]
            candidates = list(self.regression_models.keys()) if self.regression_models else canon
        else:
            canon = CLASSICAL_CLASSIFICATION if HAS_ROOT_REGISTRY else [
                'random_forest', 'logistic_regression', 'xgboost', 'gradient_boosting'
            ]
            candidates = list(self.classification_models.keys()) if self.classification_models else canon

        ordered = [m for m in canon if m in candidates]
        ordered.extend(m for m in candidates if m not in ordered and 'neural' not in m)
        if HAS_ROOT_REGISTRY:
            ordered = filter_models_by_family(ordered, model_family)
        # Drop chronically slow optional models on tiny data
        if n_samples < 200:
            ordered = [m for m in ordered if m not in ('svm', 'svr', 'knn')]
        return ordered[:8] if ordered else ['random_forest']

    def get_registry_summary(self) -> Dict[str, Any]:
        """Summarize registered models for diagnostics."""
        classification = list(self.classification_models.keys()) if self.classification_models else []
        regression = list(self.regression_models.keys()) if self.regression_models else []
        return {
            'total_models': len(set(classification + regression)),
            'classification_models': len(classification),
            'regression_models': len(regression),
            'task_types': ['classification', 'regression'],
            'sklearn_dl_models': list(SKLEARN_DL_MODELS) if HAS_ROOT_REGISTRY else [],
        }
    
    def register_model(self, model_name: str, model_class: Type[BaseEstimator],
                      task_type: str, default_params: Dict[str, Any] = None, 
                      description: str = "") -> None:
        """
        Register a new model
        
        Args:
            model_name: Name of the model
            model_class: Model class
            task_type: Task type (classification/regression)
            default_params: Default parameters
            description: Model description
        """
        if task_type == 'classification':
            self.classification_models[model_name] = model_class
        elif task_type == 'regression':
            self.regression_models[model_name] = model_class
        
        self.models[model_name] = model_class
        
        logger.info(f"✅ Registered model: {model_name} for {task_type}")


# Global model registry instance
model_registry = ModelRegistry()

# Convenience functions
def get_model(model_name: str, task_type: str = 'classification', **kwargs) -> BaseEstimator:
    """Get model from global registry"""
    return model_registry.get_model(model_name, task_type, **kwargs)

def register_model(model_name: str, model_class: Type[BaseEstimator], 
                  task_type: str, default_params: Dict[str, Any] = None, 
                  description: str = "") -> None:
    """Register model in global registry"""
    model_registry.register_model(model_name, model_class, task_type, default_params, description)

def get_available_models(task_type: str = None) -> Dict[str, List[str]]:
    """Get available models from global registry"""
    return model_registry.get_available_models(task_type)

def list_models(task_type: str = None) -> List[str]:
    """List all available model names"""
    available = model_registry.get_available_models(task_type)
    all_models = available['default'] + available['advanced']
    return sorted(list(set(all_models)))  # Remove duplicates and sort
