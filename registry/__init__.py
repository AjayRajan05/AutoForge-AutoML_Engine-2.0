"""
🧩 AutoForge Registry Module
Plugin system for features and models
"""

from .feature_registry import FeatureRegistry, FeaturePipeline, feature_registry, register_feature, get_feature, list_features
from .model_registry import ModelRegistry, model_registry, register_model, get_model, list_models

__all__ = [
    'FeatureRegistry', 
    'FeaturePipeline', 
    'feature_registry',
    'register_feature', 
    'get_feature', 
    'list_features',
    'ModelRegistry',
    'model_registry',
    'register_model',
    'get_model',
    'list_models'
]
