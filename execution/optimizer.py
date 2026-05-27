"""
⚡ Optimizer Integration - Hyperparameter optimization
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class Optimizer:
    """
    Hyperparameter optimization system
    """
    
    def __init__(self):
        """Initialize optimizer"""
        self.optimization_history = []
        self.best_params = {}
        self.best_score = None
        
    def optimize(self, model: Any, X: Any, y: Any, 
                 param_space: Dict[str, Any],
                 config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize model hyperparameters
        
        Args:
            model: Model to optimize
            X: Feature data
            y: Target data
            param_space: Parameter search space
            config: Optimization configuration
            
        Returns:
            Optimization results
        """
        try:
            logger.info("⚡ Starting hyperparameter optimization...")
            start_time = time.time()
            
            # Choose optimization strategy with configurable defaults
            try:
                from config.settings import get_config_value
                strategy = config.get('optimization_strategy', get_config_value('optimization', 'strategy', 'grid'))
                max_trials = config.get('max_trials', get_config_value('optimization', 'max_trials', 50))
                fallback_max_trials = get_config_value('optimization', 'fallback_max_trials', 20)
                combination_limit = get_config_value('optimization', 'combination_limit', 50)
            except ImportError:
                strategy = config.get('optimization_strategy', 'grid')
                max_trials = config.get('max_trials', 50)
                fallback_max_trials = 20
                combination_limit = 50
            
            if strategy == 'grid':
                results = self._grid_search(model, X, y, param_space, max_trials)
            elif strategy == 'random':
                results = self._random_search(model, X, y, param_space, max_trials)
            elif strategy == 'bayesian':
                results = self._bayesian_search(model, X, y, param_space, max_trials)
            else:
                # Fallback to simple grid search
                results = self._grid_search(model, X, y, param_space, min(max_trials, fallback_max_trials))
            
            optimization_time = time.time() - start_time
            
            # Store results
            self.optimization_history.append({
                'timestamp': time.time(),
                'strategy': strategy,
                'trials': len(results['trials']),
                'best_score': results['best_score'],
                'optimization_time': optimization_time
            })
            
            self.best_params = results['best_params']
            self.best_score = results['best_score']
            
            logger.info(f"✅ Optimization complete: {results['best_score']:.4f} score in {optimization_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}")
            return self._get_fallback_results()
    
    def _grid_search(self, model: Any, X: Any, y: Any, 
                     param_space: Dict[str, Any], max_trials: int) -> Dict[str, Any]:
        """Simple grid search implementation"""
        trials = []
        best_score = -np.inf
        best_params = {}
        
        # Generate parameter combinations (simplified)
        param_combinations = self._generate_param_combinations(param_space, max_trials, combination_limit)
        
        for i, params in enumerate(param_combinations):
            try:
                # Set parameters
                model.set_params(**params)
                
                # Simple cross-validation
                score = self._evaluate_model(model, X, y)
                
                trial_result = {
                    'trial': i,
                    'params': params,
                    'score': score,
                    'status': 'completed'
                }
                
                trials.append(trial_result)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                logger.debug(f"Trial {i+1}: {score:.4f} with params: {params}")
                
            except Exception as e:
                logger.warning(f"Trial {i+1} failed: {e}")
                trials.append({
                    'trial': i,
                    'params': params,
                    'score': None,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'trials': trials,
            'strategy': 'grid_search'
        }
    
    def _random_search(self, model: Any, X: Any, y: Any, 
                      param_space: Dict[str, Any], max_trials: int) -> Dict[str, Any]:
        """Simple random search implementation"""
        trials = []
        best_score = -np.inf
        best_params = {}
        
        for i in range(max_trials):
            try:
                # Random parameters
                params = self._sample_params(param_space)
                
                # Set parameters
                model.set_params(**params)
                
                # Evaluate
                score = self._evaluate_model(model, X, y)
                
                trial_result = {
                    'trial': i,
                    'params': params,
                    'score': score,
                    'status': 'completed'
                }
                
                trials.append(trial_result)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
            except Exception as e:
                logger.warning(f"Random trial {i+1} failed: {e}")
                trials.append({
                    'trial': i,
                    'params': params if 'params' in locals() else {},
                    'score': None,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'trials': trials,
            'strategy': 'random_search'
        }
    
    def _bayesian_search(self, model: Any, X: Any, y: Any, 
                        param_space: Dict[str, Any], max_trials: int) -> Dict[str, Any]:
        """Simplified Bayesian search (fallback to random for now)"""
        logger.info("Bayesian search not fully implemented, using random search")
        return self._random_search(model, X, y, param_space, max_trials)
    
    def _generate_param_combinations(self, param_space: Dict[str, Any], 
                                     max_trials: int, combination_limit: int = 50) -> List[Dict[str, Any]]:
        """Generate parameter combinations for grid search"""
        combinations = []
        
        # Use configurable combination limit
        try:
            from config.settings import get_config_value
            if combination_limit == 50:  # Default value, try to get from config
                combination_limit = get_config_value('optimization', 'combination_limit', 50)
        except ImportError:
            combination_limit = combination_limit
        
        # Simplified combination generation with configurable limit
        for i in range(min(max_trials, combination_limit)):
            params = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values, list):
                    # Cycle through values
                    params[param_name] = param_values[i % len(param_values)]
                else:
                    params[param_name] = param_values
            
            combinations.append(params)
        
        return combinations
    
    def _sample_params(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample random parameters"""
        params = {}
        
        for param_name, param_values in param_space.items():
            if isinstance(param_values, list):
                params[param_name] = np.random.choice(param_values)
            elif isinstance(param_values, tuple):
                # Assume (min, max, type) tuple
                if len(param_values) >= 3:
                    min_val, max_val, param_type = param_values[:3]
                    if param_type == 'int':
                        params[param_name] = np.random.randint(min_val, max_val + 1)
                    else:
                        params[param_name] = np.random.uniform(min_val, max_val)
                else:
                    params[param_name] = param_values[0]
            else:
                params[param_name] = param_values
        
        return params
    
    def _evaluate_model(self, model: Any, X: Any, y: Any) -> float:
        """Evaluate model with cross-validation"""
        try:
            from sklearn.model_selection import cross_val_score
            
            # Simple 3-fold cross-validation
            if hasattr(y, 'nunique') and y.nunique() < 20:
                # Classification
                scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            else:
                # Regression
                scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
            
            return scores.mean()
            
        except Exception as e:
            logger.warning(f"Model evaluation failed: {e}")
            return 0.0
    
    def _get_fallback_results(self) -> Dict[str, Any]:
        """Get fallback optimization results"""
        return {
            'best_params': {},
            'best_score': 0.0,
            'trials': [],
            'strategy': 'fallback',
            'error': 'Optimization failed'
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary"""
        if not self.optimization_history:
            return {'message': 'No optimization history'}
        
        latest = self.optimization_history[-1]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'latest_optimization': {
                'strategy': latest['strategy'],
                'trials': latest['trials'],
                'best_score': latest['best_score'],
                'optimization_time': latest['optimization_time']
            },
            'best_overall_score': max(opt['best_score'] for opt in self.optimization_history),
            'average_trials': np.mean([opt['trials'] for opt in self.optimization_history])
        }
