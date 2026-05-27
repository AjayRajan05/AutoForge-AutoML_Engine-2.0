"""
🧠 Meta-Optimizer - Optimization of the optimization process
Learns the best optimization strategies for different datasets
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from collections import defaultdict, Counter
from datetime import datetime

logger = logging.getLogger(__name__)


class MetaOptimizer:
    """
    Meta-Optimizer that learns the best optimization strategies
    """
    
    def __init__(self):
        """Initialize meta-optimizer"""
        self.optimization_history = []
        self.strategy_performance = defaultdict(list)
        self.dataset_optimization_patterns = {}
        
    def learn_optimization_strategy(self, dataset_profile: Dict[str, Any], 
                                  optimization_config: Dict[str, Any],
                                  final_performance: float) -> bool:
        """
        Learn from optimization results
        
        Args:
            dataset_profile: Dataset characteristics
            optimization_config: Optimization configuration used
            final_performance: Final performance score
            
        Returns:
            True if learning successful, False otherwise
            
        Raises:
            ValueError: If input validation fails
            TypeError: If input types are incorrect
        """
        # Input validation
        if dataset_profile is None:
            raise ValueError("dataset_profile cannot be None")
        
        if not isinstance(dataset_profile, dict):
            raise TypeError("dataset_profile must be a dictionary")
        
        if optimization_config is None:
            raise ValueError("optimization_config cannot be None")
        
        if not isinstance(optimization_config, dict):
            raise TypeError("optimization_config must be a dictionary")
        
        if final_performance is None or not isinstance(final_performance, (int, float)):
            raise ValueError("final_performance must be a number")
        
        logger.info("🧠 Learning optimization strategy...")
        try:
            learning_entry = {
                'dataset_profile': dataset_profile,
                'optimization_config': optimization_config,
                'final_performance': final_performance,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.optimization_history.append(learning_entry)
            
            # Extract key characteristics
            dataset_size = dataset_profile.get('size_profile', {}).get('n_samples', 1000)
            n_features = dataset_profile.get('size_profile', {}).get('n_features', 10)
            data_type = dataset_profile.get('type_profile', {}).get('primary_type', 'tabular')
            
            # Create pattern key
            pattern_key = f"{data_type}_{self._categorize_size(dataset_size)}_{self._categorize_features(n_features)}"
            
            # Store performance for this pattern
            if pattern_key not in self.dataset_optimization_patterns:
                self.dataset_optimization_patterns[pattern_key] = []
            
            self.dataset_optimization_patterns[pattern_key].append({
                'optimization_config': optimization_config,
                'performance': final_performance,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
            logger.info(f"🧠 Learned optimization strategy for {pattern_key}: performance={final_performance:.3f}")
            
        except Exception as e:
            logger.warning(f"Failed to learn optimization strategy: {e}")
    
    def recommend_optimization(self, dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimization strategy based on learned patterns"""
        try:
            # Extract characteristics
            dataset_size = dataset_profile.get('size_profile', {}).get('n_samples', 1000)
            n_features = dataset_profile.get('size_profile', {}).get('n_features', 10)
            data_type = dataset_profile.get('type_profile', {}).get('primary_type', 'tabular')
            
            # Create pattern key
            pattern_key = f"{data_type}_{self._categorize_size(dataset_size)}_{self._categorize_features(n_features)}"
            
            # Get learned patterns for this dataset type
            if pattern_key in self.dataset_optimization_patterns:
                patterns = self.dataset_optimization_patterns[pattern_key]
                
                if patterns:
                    # Find best performing optimization
                    best_pattern = max(patterns, key=lambda x: x['performance'])
                    best_config = best_pattern['optimization_config']
                    
                    # Enhance with meta-optimization insights
                    recommended_config = self._enhance_config_with_meta_insights(
                        best_config, dataset_profile, patterns
                    )
                    
                    logger.info(f"🎯 Recommended optimization for {pattern_key} based on {len(patterns)} patterns")
                    return recommended_config
            
            # Fallback to default optimization
            return self._get_default_optimization(dataset_profile)
            
        except Exception as e:
            logger.warning(f"Failed to recommend optimization: {e}")
            return self._get_default_optimization(dataset_profile)
    
    def _categorize_size(self, size: int) -> str:
        """Categorize dataset size"""
        if size < 1000:
            return "small"
        elif size < 10000:
            return "medium"
        else:
            return "large"
    
    def _categorize_features(self, n_features: int) -> str:
        """Categorize number of features"""
        if n_features < 10:
            return "low"
        elif n_features < 50:
            return "medium"
        else:
            return "high"
    
    def _enhance_config_with_meta_insights(self, base_config: Dict[str, Any], 
                                         dataset_profile: Dict[str, Any],
                                         patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhance configuration with meta-insights"""
        enhanced_config = base_config.copy()
        
        # Analyze what works best for this dataset type
        successful_configs = [p['optimization_config'] for p in patterns if p['performance'] > 0.7]
        
        if successful_configs:
            # Find common patterns in successful configurations
            common_params = self._find_common_parameters(successful_configs)
            
            # Update configuration with common successful parameters
            for param, value in common_params.items():
                enhanced_config[param] = value
            
            # Add meta-optimization specific parameters
            enhanced_config['meta_optimized'] = True
            enhanced_config['meta_confidence'] = len(successful_configs) / len(patterns)
            enhanced_config['meta_patterns_used'] = len(patterns)
        
        return enhanced_config
    
    def _find_common_parameters(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find common parameters across successful configurations"""
        common_params = {}
        
        if not configs:
            return common_params
        
        # Get all parameter keys
        all_keys = set()
        for config in configs:
            all_keys.update(config.keys())
        
        # Find parameters that are consistent across configs
        for key in all_keys:
            values = [config.get(key) for config in configs if key in config]
            if values:
                # For numeric values, find the average
                if all(isinstance(v, (int, float)) for v in values):
                    common_params[key] = np.mean(values)
                # For categorical values, find the most common
                elif all(isinstance(v, str) for v in values):
                    counter = Counter(values)
                    common_params[key] = counter.most_common(1)[0][0]
        
        return common_params
    
    def _get_default_optimization(self, dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Get default optimization configuration"""
        dataset_size = dataset_profile.get('size_profile', {}).get('n_samples', 1000)
        
        if dataset_size < 1000:
            return {
                'max_trials': 20,
                'cv_folds': 3,
                'optimization_method': 'random_search',
                'early_stopping': True,
                'meta_optimized': False
            }
        elif dataset_size < 10000:
            return {
                'max_trials': 50,
                'cv_folds': 5,
                'optimization_method': 'bayesian',
                'early_stopping': True,
                'meta_optimized': False
            }
        else:
            return {
                'max_trials': 30,
                'cv_folds': 3,
                'optimization_method': 'successive_halving',
                'early_stopping': True,
                'meta_optimized': False
            }
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights about optimization patterns"""
        try:
            insights = {
                'total_optimizations': len(self.optimization_history),
                'dataset_patterns': len(self.dataset_optimization_patterns),
                'best_performing_strategies': [],
                'optimization_trends': {}
            }
            
            # Find best performing strategies
            if self.optimization_history:
                best_strategies = sorted(
                    self.optimization_history, 
                    key=lambda x: x['final_performance'], 
                    reverse=True
                )[:5]
                
                insights['best_performing_strategies'] = [
                    {
                        'performance': entry['final_performance'],
                        'config': entry['optimization_config'],
                        'dataset_size': entry['dataset_profile'].get('size_profile', {}).get('n_samples', 0)
                    }
                    for entry in best_strategies
                ]
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get optimization insights: {e}")
            return {"error": str(e)}


# Global meta-optimizer instance
meta_optimizer = MetaOptimizer()
