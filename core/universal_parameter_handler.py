"""
Universal Parameter Handler
Handles parameter validation and processing across all AutoForge components
"""

import logging
from typing import Dict, Any, Optional, Union, List
import numpy as np

logger = logging.getLogger(__name__)

class UniversalParameterHandler:
    """
    Universal Parameter Handler
    
    Provides consistent parameter validation and processing
    across all AutoForge components
    """
    
    def __init__(self):
        """Initialize universal parameter handler"""
        self.parameter_schemas = {
            'n_estimators': {'type': int, 'min': 1, 'max': 1000, 'default': 100},
            'max_depth': {'type': int, 'min': 1, 'max': 50, 'default': 5},
            'learning_rate': {'type': float, 'min': 0.0001, 'max': 1.0, 'default': 0.1},
            'random_state': {'type': int, 'min': 0, 'max': None, 'default': 42},
            'cv_folds': {'type': int, 'min': 2, 'max': 10, 'default': 5},
            'n_trials': {'type': int, 'min': 1, 'max': 1000, 'default': 50},
            'timeout': {'type': int, 'min': 1, 'max': 3600, 'default': 300}
        }
        
    def validate_parameters(self, parameters: Dict[str, Any], 
                          schema_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate and normalize parameters
        
        Args:
            parameters: Input parameters
            schema_name: Name of parameter schema to use
            
        Returns:
            Validated and normalized parameters
        """
        try:
            validated_params = {}
            
            for param_name, param_value in parameters.items():
                if param_name in self.parameter_schemas:
                    schema = self.parameter_schemas[param_name]
                    validated_value = self._validate_single_parameter(param_name, param_value, schema)
                    validated_params[param_name] = validated_value
                else:
                    # Keep unknown parameters as-is
                    validated_params[param_name] = param_value
            
            logger.info(f"✅ Validated {len(validated_params)} parameters")
            
            return validated_params
            
        except Exception as e:
            logger.error(f"❌ Parameter validation failed: {e}")
            return parameters
    
    def _validate_single_parameter(self, name: str, value: Any, schema: Dict[str, Any]) -> Any:
        """Validate a single parameter"""
        try:
            # Type validation
            expected_type = schema['type']
            if not isinstance(value, expected_type):
                try:
                    value = expected_type(value)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert {name} to {expected_type.__name__}, using default")
                    return schema['default']
            
            # Range validation
            min_val = schema.get('min')
            max_val = schema.get('max')
            
            if min_val is not None and value < min_val:
                logger.warning(f"{name} {value} below minimum {min_val}, using minimum")
                return min_val
            
            if max_val is not None and value > max_val:
                logger.warning(f"{name} {value} above maximum {max_val}, using maximum")
                return max_val
            
            return value
            
        except Exception as e:
            logger.error(f"❌ Failed to validate {name}: {e}")
            return schema['default']
    
    def get_default_parameters(self, model_type: str) -> Dict[str, Any]:
        """
        Get default parameters for a model type
        
        Args:
            model_type: Type of model
            
        Returns:
            Default parameters
        """
        defaults = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            },
            'logistic_regression': {
                'random_state': 42,
                'max_iter': 1000
            },
            'svm': {
                'random_state': 42,
                'C': 1.0
            }
        }
        
        return defaults.get(model_type, {'random_state': 42})
    
    def merge_parameters(self, base_params: Dict[str, Any], 
                        override_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge parameter dictionaries with validation
        
        Args:
            base_params: Base parameters
            override_params: Override parameters
            
        Returns:
            Merged and validated parameters
        """
        merged = base_params.copy()
        merged.update(override_params)
        
        return self.validate_parameters(merged)

# Create global instance
universal_parameter_handler = UniversalParameterHandler()
