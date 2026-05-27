"""
🔥 BULLETPROOF ERROR HANDLER
Handle ANY error scenario gracefully
"""

import logging
import traceback
import time
from typing import Dict, Any, Optional, Tuple
from functools import wraps

logger = logging.getLogger(__name__)


class BulletproofErrorHandler:
    """
    🔥 Bulletproof Error Handler - Handle ANY error scenario gracefully
    """
    
    def __init__(self):
        """Initialize error recovery strategies"""
        self.recovery_strategies = {
            'parameter_error': self.fix_parameters,
            'data_error': self.fix_data,
            'model_error': self.switch_model,
            'memory_error': self.reduce_complexity,
            'timeout_error': self.use_faster_config,
            'import_error': self.handle_import_error,
            'validation_error': self.handle_validation_error,
            'convergence_error': self.handle_convergence_error,
            'compatibility_error': self.handle_compatibility_error,
        }
        
        # Fallback configurations
        self.fallback_configs = {
            'minimal': {
                'n_trials': 1,
                'cv_folds': 2,
                'timeout': 30,
                'simple_models_only': True
            },
            'safe': {
                'n_trials': 3,
                'cv_folds': 3,
                'timeout': 60,
                'simple_models_only': False
            },
            'standard': {
                'n_trials': 10,
                'cv_folds': 5,
                'timeout': 120,
                'simple_models_only': False
            }
        }
        
        # Error statistics
        self.error_counts = {}
        self.recovery_success = {}
    
    def classify_error(self, error: Exception) -> str:
        """Classify error type for appropriate recovery"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Parameter-related errors
        if any(keyword in error_str for keyword in ['parameter', 'solver', 'optimizer', 'invalid']):
            return 'parameter_error'
        
        # Data-related errors
        elif any(keyword in error_str for keyword in ['data', 'shape', 'type', 'nan', 'inf']):
            return 'data_error'
        
        # Model-related errors
        elif any(keyword in error_str for keyword in ['model', 'fit', 'predict', 'estimator']):
            return 'model_error'
        
        # Memory-related errors
        elif any(keyword in error_str for keyword in ['memory', 'ram', 'allocation']):
            return 'memory_error'
        
        # Timeout-related errors
        elif any(keyword in error_str for keyword in ['timeout', 'time', 'limit']):
            return 'timeout_error'
        
        # Import-related errors
        elif error_type in ['importerror', 'modulenotfounderror']:
            return 'import_error'
        
        # Validation-related errors
        elif error_type in ['valueerror', 'typeerror']:
            return 'validation_error'
        
        # Convergence-related errors
        elif any(keyword in error_str for keyword in ['convergence', 'converge', 'iteration']):
            return 'convergence_error'
        
        # Compatibility-related errors
        elif any(keyword in error_str for keyword in ['compatible', 'version', 'platform']):
            return 'compatibility_error'
        
        else:
            return 'unknown_error'
    
    def handle_any_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Universal error recovery"""
        # Classify error
        error_type = self.classify_error(error)
        
        # Track error statistics
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        logger.error(f"Error classified as: {error_type}")
        logger.error(f"Error details: {str(error)}")
        
        # Get recovery strategy
        recovery_strategy = self.recovery_strategies.get(error_type, self.generic_recovery)
        
        try:
            # Apply recovery strategy
            recovery_info = recovery_strategy(error, context)
            
            # Track recovery success
            self.recovery_success[error_type] = self.recovery_success.get(error_type, 0) + 1
            
            logger.info(f"Successfully applied recovery strategy for {error_type}")
            return True, recovery_info
            
        except Exception as recovery_error:
            logger.error(f"Recovery strategy failed: {str(recovery_error)}")
            
            # Apply generic recovery
            try:
                generic_info = self.generic_recovery(error, context)
                return True, generic_info
            except Exception as generic_error:
                logger.error(f"Generic recovery also failed: {str(generic_error)}")
                return False, {'error': str(error), 'recovery_failed': True}
    
    def fix_parameters(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fix parameter-related errors"""
        logger.info("Applying parameter fix...")
        
        # Get current parameters
        current_params = context.get('parameters', {})
        
        # Simple parameter fixes
        model_name = context.get('model_name', 'unknown')
        task_type = context.get('task_type', 'classification')
        
        # Apply universal parameter fixes
        fixed_params = self._validate_and_correct_parameters(model_name, current_params, task_type)
        
        return {
            'parameters': fixed_params,
            'recovery_action': 'parameter_fix',
            'original_error': str(error)
        }
    
    def fix_data(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fix data-related errors"""
        logger.info("Applying data fix...")
        
        # Get data
        X = context.get('X')
        y = context.get('y')
        
        if X is None:
            return {'error': 'No data available for fixing'}
        
        import pandas as pd
        import numpy as np
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Fix missing values
        if X.isnull().any().any():
            X = X.fillna(X.mean())
        
        # Fix infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Fix data types
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    X[col] = X[col].astype('category').cat.codes
        
        return {
            'X': X,
            'y': y,
            'recovery_action': 'data_fix',
            'original_error': str(error)
        }
    
    def switch_model(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Switch to a different model"""
        logger.info("Switching to fallback model...")
        
        # Simple fallback models
        fallback_models = {
            'classification': ['logistic_regression', 'naive_bayes'],
            'regression': ['linear_regression', 'ridge']
        }
        
        task_type = context.get('task_type', 'classification')
        current_model = context.get('model_name', '')
        method_name = context.get('method', '')
        
        # Get fallback models
        models_to_try = fallback_models.get(task_type, ['logistic_regression'])
        
        # For fit/predict methods, don't return model_name - it's not a valid parameter
        if method_name in ['fit', 'predict', 'predict_proba']:
            # Just return a simple recovery without model_name
            return {
                'recovery_action': 'model_switch',
                'original_error': str(error),
                'message': 'Model switch recovery applied'
            }
        
        # For other methods, return model_name
        for model_name in models_to_try:
            if model_name != current_model:
                logger.info(f"Switching to fallback model: {model_name}")
                return {
                    'model_name': model_name,
                    'recovery_action': 'model_switch',
                    'original_error': str(error)
                }
        
        return {'error': 'No suitable fallback model found'}
    
    #def reduce_complexity(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
    def reduce_complexity(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce complexity for memory errors"""
        logger.info("Reducing complexity...")
        
        # Get current configuration
        current_config = context.get('config', {})
        
        # Apply minimal configuration
        minimal_config = self.fallback_configs['minimal'].copy()
        
        # Override with minimal settings
        minimal_config.update({
            'max_depth': 3,
            'n_estimators': 10,
            'max_features': 'sqrt',
            'subsample': 0.5
        })
        
        return {
            'config': minimal_config,
            'recovery_action': 'complexity_reduction',
            'original_error': str(error)
        }
    
    def use_faster_config(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use faster configuration for timeout errors"""
        logger.info("Applying faster configuration...")
        
        # Get current configuration
        current_config = context.get('config', {})
        
        # Apply safe configuration with reduced trials
        safe_config = self.fallback_configs['safe'].copy()
        safe_config.update({
            'timeout': 30,  # 30 seconds max
            'early_stopping': True,
            'fast_models_only': True
        })
        
        return {
            'config': safe_config,
            'recovery_action': 'speed_optimization',
            'original_error': str(error)
        }
    
    def handle_import_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle import errors"""
        logger.info("Handling import error...")
        
        # Try to install missing dependencies or use alternatives
        error_str = str(error).lower()
        
        # Check for specific missing modules
        if 'lightgbm' in error_str:
            return {
                'disable_lightgbm': True,
                'recovery_action': 'disable_feature',
                'original_error': str(error)
            }
        elif 'xgboost' in error_str:
            return {
                'disable_xgboost': True,
                'recovery_action': 'disable_feature',
                'original_error': str(error)
            }
        else:
            return {
                'disable_advanced_features': True,
                'recovery_action': 'disable_features',
                'original_error': str(error)
            }
    
    def handle_validation_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle validation errors"""
        logger.info("Handling validation error...")
        
        # Apply parameter validation fix
        return self.fix_parameters(error, context)
    
    def handle_convergence_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle convergence errors"""
        logger.info("Handling convergence error...")
        
        # Increase max iterations and adjust tolerance
        current_params = context.get('parameters', {})
        
        fixed_params = current_params.copy()
        fixed_params.update({
            'max_iter': current_params.get('max_iter', 100) * 2,
            'tol': current_params.get('tol', 1e-4) * 10
        })
        
        return {
            'parameters': fixed_params,
            'recovery_action': 'convergence_fix',
            'original_error': str(error)
        }
    
    def handle_compatibility_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle compatibility errors"""
        logger.info("Handling compatibility error...")
        
        # Apply compatibility fixes
        return {
            'config': self.fallback_configs['minimal'],
            'disable_advanced_features': True,
            'recovery_action': 'compatibility_fix',
            'original_error': str(error)
        }
    
    def generic_recovery(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generic recovery strategy"""
        logger.info("Applying generic recovery...")
        
        method_name = context.get('method', '')
        
        # Apply minimal configuration as last resort
        minimal_config = self.fallback_configs['minimal'].copy()
        
        # For fit/predict methods, don't return model_name
        if method_name in ['fit', 'predict', 'predict_proba']:
            return {
                'config': minimal_config,
                'recovery_action': 'generic_recovery',
                'original_error': str(error),
                'message': 'Generic recovery applied'
            }
        
        return {
            'config': minimal_config,
            'model_name': 'logistic_regression',  # Most reliable model
            'recovery_action': 'generic_recovery',
            'original_error': str(error)
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics"""
        return {
            'error_counts': self.error_counts,
            'recovery_success': self.recovery_success,
            'total_errors': sum(self.error_counts.values()),
            'total_recoveries': sum(self.recovery_success.values())
        }
    
    def _validate_and_correct_parameters(self, model_name: str, params: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Simple parameter validation and correction"""
        corrected = params.copy()
        
        # Common parameter fixes
        if 'n_estimators' in corrected:
            corrected['n_estimators'] = max(1, min(corrected['n_estimators'], 1000))
        
        if 'max_depth' in corrected:
            corrected['max_depth'] = max(1, min(corrected['max_depth'], 20))
        
        if 'learning_rate' in corrected:
            corrected['learning_rate'] = max(0.0001, min(corrected['learning_rate'], 1.0))
        
        if 'random_state' not in corrected:
            corrected['random_state'] = 42
        
        return corrected


def bulletproof_method(max_retries: int = 3):
    """Decorator to make any method bulletproof"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = BulletproofErrorHandler()
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        # Last attempt failed
                        logger.error(f"Method {func.__name__} failed after {max_retries} retries")
                        raise
                    
                    # Try to recover
                    context = {
                        'method': func.__name__,
                        'args': args,
                        'kwargs': kwargs,
                        'attempt': attempt + 1
                    }
                    
                    recovered, recovery_info = error_handler.handle_any_error(e, context)
                    
                    if recovered:
                        logger.info(f"Recovery successful for {func.__name__}, retrying...")
                        
                        # Apply recovery info carefully
                        if 'config' in recovery_info:
                            # Update config on the object if it has one
                            if hasattr(args[0], 'config'):
                                args[0].config.update(recovery_info['config'])
                        
                        # Don't pass model_name to fit/predict methods
                        if func.__name__ in ['fit', 'predict', 'predict_proba']:
                            # Filter out invalid kwargs for these methods
                            valid_kwargs = {}
                            for key, value in kwargs.items():
                                if key not in ['model_name', 'parameters']:
                                    valid_kwargs[key] = value
                            kwargs = valid_kwargs
                        
                        time.sleep(1)  # Brief pause before retry
                        continue
                    else:
                        logger.error(f"Recovery failed for {func.__name__}")
                        raise
            
            return None
        return wrapper
    return decorator


# Global instance
bulletproof_error_handler = BulletproofErrorHandler()
