"""
🎯 Strategy Selector - Intelligent strategy selection based on dataset characteristics
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from features.meta_learning.pattern_learner import PatternLearner
from features.meta_learning.knowledge_base import KnowledgeBase
from .meta_optimizer import MetaOptimizer

logger = logging.getLogger(__name__)


class StrategySelector:
    """
    Intelligent Strategy Selector
    
    Selects optimal AutoML strategies based on dataset characteristics,
    past experience, and machine learning rules.
    """
    
    def __init__(self):
        """Initialize strategy selector"""
        self.rules = self._load_strategy_rules()
        self.knowledge_base = KnowledgeBase()
        self.pattern_learner = PatternLearner()
        self.meta_optimizer = MetaOptimizer()
        self.strategy_history = []
        
    def select_strategy(self, dataset_profile: Dict[str, Any], 
                      user_preference: str = "auto",
                      max_time: float = None,
                      max_trials: int = None) -> Dict[str, Any]:
        """
        Select optimal strategy based on dataset characteristics
        
        Args:
            dataset_profile: Dataset analysis results
            user_preference: User strategy preference ("auto", "fast", "accurate", "robust")
            max_time: Maximum time constraint
            max_trials: Maximum trials constraint
            
        Returns:
            Selected strategy dictionary
        """
        try:
            logger.info(f"🎯 Selecting strategy for {dataset_profile.get('size_profile', {}).get('category', 'unknown')} dataset")
            
            # Step 1: Apply rule-based selection
            base_strategy = self._apply_rules(dataset_profile, user_preference)
            
            # Step 2: Enhance with meta-learning
            learned_strategy = self.knowledge_base.get_similar_strategy(dataset_profile)
            
            # Step 3: Apply constraints
            constrained_strategy = self._apply_constraints(
                base_strategy, max_time, max_trials
            )
            
            # Step 4: Merge strategies
            final_strategy = self._merge_strategies(
                constrained_strategy, learned_strategy
            )
            
            # Step 3: Apply meta-learning enhancement
            logger.info("🧠 Applying meta-learning enhancement...")
            enhanced_recommendations = self._apply_meta_learning_enhancement(
                dataset_profile, final_strategy, user_preference
            )
            
            # Step 4: Apply constraints and user preferences
            logger.info("⚖️  Applying constraints and user preferences...")
            final_recommendations = self._apply_constraints(
                enhanced_recommendations, max_time, max_trials, user_preference
            )
            
            # Step 5: Apply meta-optimization
            logger.info("🧠 Applying meta-optimization...")
            optimization_config = self.meta_optimizer.recommend_optimization(dataset_profile)
            final_recommendations['optimization'] = optimization_config
            
            # Step 6: Learn from this selection
            self._queue_learning(dataset_profile, final_recommendations)
            
            # Store strategy history
            self.strategy_history.append({
                'dataset_profile': dataset_profile,
                'recommendations': final_recommendations,
                'user_preference': user_preference,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            # Add metadata
            final_recommendations["metadata"] = {
                "selection_time": pd.Timestamp.now().isoformat(),
                "user_preference": user_preference,
                "dataset_profile": dataset_profile,
                "confidence": self._calculate_confidence(final_recommendations, dataset_profile),
                "primary_strategy": self._determine_primary_strategy(final_recommendations)
            }
            
            logger.info(f"✅ Strategy selected: {final_recommendations['metadata']['primary_strategy']}")
            return final_recommendations
            
        except Exception as e:
            logger.error(f"❌ Strategy selection failed: {e}")
            return self._get_fallback_strategy()
    
    def learn_from_result(self, dataset_profile: Dict[str, Any], 
                        strategy: Dict[str, Any], performance: float):
        """
        Learn from execution results for future improvements
        
        Args:
            dataset_profile: Dataset characteristics
            strategy: Strategy that was used
            performance: Performance achieved (0-1)
        """
        try:
            self.knowledge_base.learn_from_result(
                dataset_profile, strategy, performance
            )
            logger.info(f"🧠 Learned from execution: performance={performance:.3f}")
        except Exception as e:
            logger.error(f"❌ Learning failed: {e}")
    
    def _apply_rules(self, profile: Dict[str, Any], preference: str) -> Dict[str, Any]:
        """Apply rule-based strategy selection"""
        strategy = {
            "preprocessing": [],
            "feature_engineering": [],
            "models": [],
            "optimization": {},
            "validation": {}
        }
        
        # Size-based rules
        size_profile = profile.get("size_profile", {})
        size_category = size_profile.get("category", "medium")
        
        if size_category == "large":
            strategy["preprocessing"].extend(["sampling", "incremental"])
            strategy["optimization"]["max_trials"] = 50
            strategy["models"] = ["lightgbm", "xgboost", "random_forest"]
            strategy["validation"]["cv_folds"] = 3
            
        elif size_category == "small":
            strategy["validation"]["cv_folds"] = 5
            strategy["optimization"]["max_trials"] = 100
            strategy["models"] = ["random_forest", "svm", "logistic_regression", "neural_network"]
            
        else:  # medium
            strategy["validation"]["cv_folds"] = 5
            strategy["optimization"]["max_trials"] = 75
            strategy["models"] = ["xgboost", "lightgbm", "random_forest", "logistic_regression"]
        
        # Quality-based rules
        quality_profile = profile.get("quality_profile", {})
        quality_category = quality_profile.get("category", "good")
        
        if quality_category == "poor":
            strategy["preprocessing"].extend(["robust_imputation", "outlier_removal"])
            strategy["models"] = ["random_forest", "xgboost"]  # Robust models
            
        elif quality_category == "moderate":
            strategy["preprocessing"].append("simple_imputation")
        
        # Type-based rules
        type_profile = profile.get("type_profile", {})
        
        if type_profile.get("has_categorical", False):
            strategy["preprocessing"].append("categorical_encoding")
        
        if type_profile.get("has_mixed_types", False):
            strategy["preprocessing"].append("type_standardization")
        
        # Complexity-based rules
        complexity_profile = profile.get("complexity_profile", {})
        
        if complexity_profile.get("is_imbalanced", False):
            strategy["preprocessing"].append("class_balancing")
            strategy["models"] = [m for m in strategy["models"] if m in ["xgboost", "lightgbm", "random_forest"]]
        
        if complexity_profile.get("is_high_stress", False):
            strategy["feature_engineering"].append("dimensionality_reduction")
            strategy["optimization"]["max_trials"] = min(strategy["optimization"]["max_trials"], 30)
        
        # Preference-based adjustments
        if preference == "fast":
            strategy["optimization"]["max_trials"] = min(strategy["optimization"]["max_trials"], 20)
            strategy["models"] = strategy["models"][:2]  # Use top 2 fastest models
            
        elif preference == "accurate":
            strategy["optimization"]["max_trials"] = strategy["optimization"]["max_trials"] * 2
            strategy["validation"]["cv_folds"] = min(strategy["validation"]["cv_folds"] + 2, 10)
            
        elif preference == "robust":
            strategy["preprocessing"].extend(["robust_imputation", "outlier_removal"])
            strategy["models"] = ["random_forest", "xgboost"]  # Most robust models
        
        return strategy
    
    def _apply_constraints(self, strategy: Dict[str, Any], 
                        max_time: float, max_trials: int, 
                        user_preference: str = "auto") -> Dict[str, Any]:
        """Apply user constraints to strategy"""
        constrained_strategy = strategy.copy()
        
        # Time constraint
        if max_time:
            # Estimate time per trial and adjust
            current_trials = constrained_strategy["optimization"].get("max_trials", 50)
            if max_time < 60:  # Less than 1 minute
                constrained_strategy["optimization"]["max_trials"] = min(current_trials, 10)
                constrained_strategy["preprocessing"].append("fast_mode")
            elif max_time < 300:  # Less than 5 minutes
                constrained_strategy["optimization"]["max_trials"] = min(current_trials, 30)
        
        # Trial constraint
        if max_trials:
            constrained_strategy["optimization"]["max_trials"] = min(
                constrained_strategy["optimization"]["max_trials"], 
                max_trials
            )
        
        # User preference constraint
        if user_preference != "auto":
            preference_rules = self.strategy_templates["preference_rules"].get(user_preference, {})
            
            # Apply time multiplier
            if "time_multiplier" in preference_rules and max_time:
                adjusted_time = max_time * preference_rules["time_multiplier"]
                # Recalculate trials based on adjusted time
                if adjusted_time < 60:
                    constrained_strategy["optimization"]["max_trials"] = 5
                elif adjusted_time < 300:
                    constrained_strategy["optimization"]["max_trials"] = 20
            
            # Apply model limit
            if "model_limit" in preference_rules:
                current_models = constrained_strategy.get("models", [])
                constrained_strategy["models"] = current_models[:preference_rules["model_limit"]]
            
            # Apply preferred models
            if "models" in preference_rules:
                preferred_models = preference_rules["models"]
                constrained_strategy["models"] = preferred_models
        
        return constrained_strategy
    
    def _merge_strategies(self, base_strategy: Dict[str, Any], 
                        learned_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Merge rule-based and learned strategies"""
        merged = base_strategy.copy()
        
        if not learned_strategy:
            return merged
        
        # Merge preprocessing steps
        learned_preprocessing = learned_strategy.get("preprocessing", [])
        for step in learned_preprocessing:
            if step not in merged["preprocessing"]:
                merged["preprocessing"].append(step)
        
        # Merge models (prioritize learned if available)
        learned_models = learned_strategy.get("models", [])
        if learned_models:
            # Use learned models but keep some base models for diversity
            merged["models"] = learned_models + [
                m for m in base_strategy.get("models", []) 
                if m not in learned_models
            ][:2]
        
        # Merge optimization parameters
        learned_opt = learned_strategy.get("optimization", {})
        for key, value in learned_opt.items():
            if key not in merged["optimization"]:
                merged["optimization"][key] = value
        
        # Add learning confidence
        merged["learning_confidence"] = learned_strategy.get("confidence", 0.5)
        
        return merged
    
    def _calculate_confidence(self, strategy: Dict[str, Any], 
                           profile: Dict[str, Any]) -> float:
        """Calculate confidence in strategy selection"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for clear cases
        size_profile = profile.get("size_profile", {})
        if size_profile.get("category") in ["small", "large"]:
            confidence += 0.2
        
        quality_profile = profile.get("quality_profile", {})
        if quality_profile.get("category") == "good":
            confidence += 0.2
        
        # Increase confidence if we have learning data
        if strategy.get("learning_confidence", 0) > 0:
            confidence += 0.2
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def _determine_primary_strategy(self, strategy: Dict[str, Any]) -> str:
        """Determine primary strategy type"""
        models = strategy.get("models", [])
        
        if "neural_network" in models:
            return "deep_learning"
        elif any(m in models for m in ["xgboost", "lightgbm"]):
            return "gradient_boosting"
        elif "random_forest" in models:
            return "ensemble"
        elif "svm" in models:
            return "kernel_methods"
        else:
            return "tree_based"
    
    def _get_fallback_strategy(self) -> Dict[str, Any]:
        """Get fallback strategy when selection fails"""
        return {
            "preprocessing": ["simple_imputation"],
            "feature_engineering": [],
            "models": ["random_forest"],
            "optimization": {"max_trials": 20},
            "validation": {"cv_folds": 3},
            "metadata": {
                "selection_time": datetime.now().isoformat(),
                "fallback": True,
                "confidence": 0.3,
                "primary_strategy": "fallback"
            }
        }
    
    def _load_strategy_rules(self) -> Dict[str, Any]:
        """Load strategy rules (could be from file in future)"""
        # Use configurable strategy rules
        try:
            from config.settings import get_config_value
            return {
                "size_rules": {
                    "small": {
                        "max_trials": get_config_value('strategy', 'small_max_trials', 100), 
                        "cv_folds": get_config_value('strategy', 'small_cv_folds', 5)
                    },
                    "medium": {
                        "max_trials": get_config_value('strategy', 'medium_max_trials', 75), 
                        "cv_folds": get_config_value('strategy', 'medium_cv_folds', 5)
                    },
                    "large": {
                        "max_trials": get_config_value('strategy', 'large_max_trials', 50), 
                        "cv_folds": get_config_value('strategy', 'large_cv_folds', 3)
                    }
                },
                "quality_rules": {
                    "good": {"models": get_config_value('strategy', 'good_quality_models', ["xgboost", "lightgbm", "random_forest"])},
                    "moderate": {
                        "models": get_config_value('strategy', 'moderate_quality_models', ["random_forest", "xgboost"]), 
                        "preprocessing": get_config_value('strategy', 'moderate_preprocessing', ["simple_imputation"])
                    },
                    "poor": {
                        "models": get_config_value('strategy', 'poor_quality_models', ["random_forest"]), 
                        "preprocessing": get_config_value('strategy', 'poor_preprocessing', ["robust_imputation", "outlier_removal"])
                    }
                },
                "preference_rules": {
                    "fast": {
                        "time_multiplier": get_config_value('strategy', 'fast_time_multiplier', 0.3), 
                        "model_limit": get_config_value('strategy', 'fast_model_limit', 2)
                    },
                    "accurate": {
                        "time_multiplier": get_config_value('strategy', 'accurate_time_multiplier', 2.0), 
                        "cv_multiplier": get_config_value('strategy', 'accurate_cv_multiplier', 1.5)
                    },
                    "robust": {
                        "models": get_config_value('strategy', 'robust_models', ["random_forest", "xgboost"]), 
                        "preprocessing": get_config_value('strategy', 'robust_preprocessing', ["robust_imputation"])
                    }
                }
            }
        except ImportError:
            # Fallback to hardcoded values
            return {
                "size_rules": {
                    "small": {"max_trials": 100, "cv_folds": 5},
                    "medium": {"max_trials": 75, "cv_folds": 5},
                    "large": {"max_trials": 50, "cv_folds": 3}
                },
                "quality_rules": {
                    "good": {"models": ["xgboost", "lightgbm", "random_forest"]},
                    "moderate": {"models": ["random_forest", "xgboost"], "preprocessing": ["simple_imputation"]},
                    "poor": {"models": ["random_forest"], "preprocessing": ["robust_imputation", "outlier_removal"]}
                },
                "preference_rules": {
                    "fast": {"time_multiplier": 0.3, "model_limit": 2},
                    "accurate": {"time_multiplier": 2.0, "cv_multiplier": 1.5},
                    "robust": {"models": ["random_forest", "xgboost"], "preprocessing": ["robust_imputation"]}
                }
            }
    
    def _apply_meta_learning_enhancement(self, dataset_profile: Dict[str, Any], 
                                       recommendations: Dict[str, Any],
                                       user_preference: str) -> Dict[str, Any]:
        """Apply meta-learning enhancement to recommendations"""
        try:
            # Use pattern learner to find similar datasets
            similar_patterns = self.pattern_learner.find_similar_patterns(dataset_profile)
            
            if similar_patterns:
                logger.info(f"🧠 Found {len(similar_patterns)} similar patterns")
                
                # Enhance recommendations based on patterns
                enhanced_recommendations = recommendations.copy()
                
                # Adjust model recommendations based on patterns
                pattern_models = []
                for pattern in similar_patterns[:3]:  # Top 3 patterns
                    if 'best_model' in pattern:
                        pattern_models.append(pattern['best_model'])
                
                if pattern_models:
                    # Add pattern-based models to recommendations
                    existing_models = enhanced_recommendations.get('models', [])
                    for model in pattern_models:
                        if model not in existing_models:
                            existing_models.insert(0, model)  # Prioritize pattern models
                    enhanced_recommendations['models'] = existing_models[:5]  # Keep top 5
                
                return enhanced_recommendations
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"⚠️ Meta-learning enhancement failed: {e}")
            return recommendations
    
    def _learn_from_selection(self, dataset_profile: Dict[str, Any], 
                             selected_strategy: Dict[str, Any],
                             performance_score: float):
        """Learn from strategy selection for future improvements"""
        try:
            # Store experiment in knowledge base
            experiment = {
                'dataset_profile': dataset_profile,
                'strategy': selected_strategy,
                'performance': performance_score,
                'timestamp': datetime.now().isoformat()
            }
            
            self.knowledge_base.add_experiment(experiment)
            
            # Update pattern learner with new insights
            pattern_data = {
                'dataset_profile': dataset_profile,
                'strategy': selected_strategy,
                'performance': performance_score,
                'timestamp': datetime.now().isoformat()
            }
            self.pattern_learner.learn_from_experiment(pattern_data)
            
            # Update strategy rules based on performance
            self._update_strategy_rules(dataset_profile, selected_strategy, performance_score)
            
            logger.info(f"🧠 Learned from selection: performance={performance_score:.3f}")
            
        except Exception as e:
            logger.warning(f"⚠️ Learning failed: {e}")
    
    def _update_strategy_rules(self, dataset_profile: Dict[str, Any], 
                              strategy: Dict[str, Any], performance: float):
        """Update strategy rules based on performance feedback"""
        try:
            # Use configurable thresholds
            try:
                from config.settings import get_config_value
                good_performance_threshold = get_config_value('strategy', 'good_performance_threshold', 0.8)
                large_dataset_threshold = get_config_value('strategy', 'large_dataset_threshold', 10000)
                small_dataset_threshold = get_config_value('strategy', 'small_dataset_threshold', 1000)
                high_feature_threshold = get_config_value('strategy', 'high_feature_threshold', 50)
                max_models_to_keep = get_config_value('strategy', 'max_models_to_keep', 5)
            except ImportError:
                good_performance_threshold = 0.8
                large_dataset_threshold = 10000
                small_dataset_threshold = 1000
                high_feature_threshold = 50
                max_models_to_keep = 5
            
            # Extract key characteristics
            dataset_size = dataset_profile.get('size_profile', {}).get('n_samples', 1000)
            n_features = dataset_profile.get('size_profile', {}).get('n_features', 10)
            
            # Update performance-based rules
            if performance > good_performance_threshold:  # Good performance
                # Store successful strategy pattern
                if dataset_size > large_dataset_threshold:
                    self.strategy_templates['size_rules']['large']['best_models'] = strategy.get('models', ['random_forest'])
                elif dataset_size < small_dataset_threshold:
                    self.strategy_templates['size_rules']['small']['best_models'] = strategy.get('models', ['random_forest'])
                
                if n_features > high_feature_threshold:
                    self.strategy_templates['feature_rules']['high_dimensional']['best_models'] = strategy.get('models', ['xgboost'])
                
                # Update model list limit
                existing_models = strategy.get('models', [])
                for model in pattern_models:
                    if model not in existing_models:
                        existing_models.insert(0, model)  # Prioritize pattern models
                strategy['models'] = existing_models[:max_models_to_keep]  # Keep configurable limit
                
            # Store in knowledge base for future reference
            learning_entry = {
                'dataset_size': dataset_size,
                'n_features': n_features,
                'strategy': strategy,
                'performance': performance,
                'timestamp': datetime.now().isoformat()
            }
            
            self.knowledge_base.add_experiment(learning_entry)
            
        except Exception as e:
            logger.warning(f"Failed to update strategy rules: {e}")
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get insights from learned performance patterns"""
        try:
            experiments = self.knowledge_base.get_experiments()
            
            if not experiments:
                return {"message": "No performance data available"}
            
            # Analyze performance patterns
            insights = {
                'total_experiments': len(experiments),
                'average_performance': np.mean([exp.get('performance', 0) for exp in experiments]),
                'best_performing_strategies': [],
                'dataset_size_patterns': {},
                'feature_patterns': {}
            }
            
            # Find best performing strategies
            sorted_experiments = sorted(experiments, key=lambda x: x.get('performance', 0), reverse=True)
            insights['best_performing_strategies'] = [
                {
                    'strategy': exp.get('strategy', {}),
                    'performance': exp.get('performance', 0),
                    'dataset_size': exp.get('dataset_profile', {}).get('size_profile', {}).get('n_samples', 0)
                }
                for exp in sorted_experiments[:5]
            ]
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get performance insights: {e}")
            return {"error": str(e)}
    
    def _queue_learning(self, dataset_profile: Dict[str, Any], recommendations: Dict[str, Any]):
        """Queue learning for when performance results are available"""
        # Store for later learning when performance is known
        self._pending_learning = {
            'dataset_profile': dataset_profile,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
    
    def complete_learning_cycle(self, performance_score: float):
        """Complete the learning cycle with performance results"""
        try:
            if hasattr(self, '_pending_learning'):
                # Learn from strategy selection
                self._learn_from_selection(
                    self._pending_learning['dataset_profile'],
                    self._pending_learning['recommendations'],
                    performance_score
                )
                
                # Learn optimization strategy
                self.meta_optimizer.learn_optimization_strategy(
                    self._pending_learning['dataset_profile'],
                    self._pending_learning['recommendations'].get('optimization', {}),
                    performance_score
                )
                
                logger.info(f"🧠 Completed learning cycle: performance={performance_score:.3f}")
                delattr(self, '_pending_learning')
                
        except Exception as e:
            logger.warning(f"Failed to complete learning cycle: {e}")


class KnowledgeBase:
    """
    Knowledge Base for Meta-Learning
    
    Stores and learns from past AutoML executions to improve
    future strategy selections.
    """
    
    def __init__(self):
        self.experiments = []
        self.patterns = []
        self.load_experiments()
    
    def get_similar_strategy(self, dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Get strategy from similar past experiments"""
        if not self.experiments:
            return {}
        
        # Find similar experiments
        similar_experiments = self._find_similar_experiments(dataset_profile)
        
        if not similar_experiments:
            return {}
        
        # Return best performing strategy from similar cases
        best_experiment = max(similar_experiments, key=lambda x: x.get("performance", 0))
        
        return {
            "models": best_experiment.get("strategy", {}).get("models", []),
            "preprocessing": best_experiment.get("strategy", {}).get("preprocessing", []),
            "optimization": best_experiment.get("strategy", {}).get("optimization", {}),
            "confidence": self._calculate_similarity(best_experiment, dataset_profile),
            "source": "meta_learning"
        }
    
    def learn_from_result(self, dataset_profile: Dict[str, Any], 
                        strategy: Dict[str, Any], performance: float):
        """Learn from execution results"""
        experiment = {
            "id": len(self.experiments),
            "dataset_profile": dataset_profile,
            "strategy": strategy,
            "performance": performance,
            "timestamp": datetime.now().isoformat()
        }
        
        self.experiments.append(experiment)
        self.save_experiments()
        
        # Extract patterns periodically
        if len(self.experiments) % 10 == 0:
            self._extract_patterns()
    
    def _find_similar_experiments(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find experiments with similar dataset profiles"""
        similar = []
        
        for exp in self.experiments:
            similarity = self._calculate_similarity(exp, profile)
            if similarity > 0.7:  # High similarity threshold
                similar.append(exp)
        
        return similar
    
    def _calculate_similarity(self, experiment: Dict[str, Any], 
                           profile: Dict[str, Any]) -> float:
        """Calculate similarity between experiment and current profile"""
        exp_profile = experiment.get("dataset_profile", {})
        
        # Size similarity
        exp_size = exp_profile.get("size_profile", {}).get("category", "medium")
        cur_size = profile.get("size_profile", {}).get("category", "medium")
        size_sim = 1.0 if exp_size == cur_size else 0.5
        
        # Quality similarity
        exp_quality = exp_profile.get("quality_profile", {}).get("category", "good")
        cur_quality = profile.get("quality_profile", {}).get("category", "good")
        quality_sim = 1.0 if exp_quality == cur_quality else 0.5
        
        # Type similarity
        exp_has_cat = exp_profile.get("type_profile", {}).get("has_categorical", False)
        cur_has_cat = profile.get("type_profile", {}).get("has_categorical", False)
        type_sim = 1.0 if exp_has_cat == cur_has_cat else 0.7
        
        # Weighted average
        return (size_sim * 0.4 + quality_sim * 0.3 + type_sim * 0.3)
    
    def _extract_patterns(self):
        """Extract patterns from experiments"""
        # Group by dataset characteristics
        patterns = {}
        
        for exp in self.experiments:
            # Create pattern key
            size = exp.get("dataset_profile", {}).get("size_profile", {}).get("category", "unknown")
            quality = exp.get("dataset_profile", {}).get("quality_profile", {}).get("category", "unknown")
            key = f"{size}_{quality}"
            
            if key not in patterns:
                patterns[key] = []
            
            patterns[key].append(exp)
        
        # Find best strategies for each pattern
        self.patterns = []
        for pattern_key, exps in patterns.items():
            if len(exps) >= 3:  # Need at least 3 examples
                best_exp = max(exps, key=lambda x: x.get("performance", 0))
                self.patterns.append({
                    "pattern": pattern_key,
                    "best_strategy": best_exp.get("strategy", {}),
                    "avg_performance": sum(e.get("performance", 0) for e in exps) / len(exps),
                    "count": len(exps)
                })
    
    def save_experiments(self):
        """Save experiments to file"""
        try:
            # Use configurable experiment history limit
            try:
                from config.settings import get_config_value
                max_experiments_to_keep = get_config_value('strategy', 'max_experiments_to_keep', 100)
            except ImportError:
                max_experiments_to_keep = 100
            
            with open("knowledge_base.json", "w") as f:
                json.dump(self.experiments[-max_experiments_to_keep:], f, indent=2, default=str)  # Keep configurable limit
        except Exception as e:
            logger.error(f"Failed to save experiments: {e}")
    
    def _apply_meta_learning_enhancement(self, dataset_profile: Dict[str, Any], 
                                       recommendations: Dict[str, Any],
                                       user_preference: str) -> Dict[str, Any]:
        """Apply meta-learning enhancement to recommendations"""
        try:
            # Use pattern learner to find similar datasets
            similar_patterns = self.pattern_learner.find_similar_patterns(dataset_profile)
            
            if similar_patterns:
                logger.info(f"🧠 Found {len(similar_patterns)} similar patterns")
                
                # Enhance recommendations based on patterns
                enhanced_recommendations = recommendations.copy()
                
                # Adjust model recommendations based on patterns
                pattern_models = []
                for pattern in similar_patterns:
                    if 'best_models' in pattern:
                        pattern_models.extend(pattern['best_models'])
                
                if pattern_models:
                    # Count model frequencies
                    from collections import Counter
                    model_counts = Counter(pattern_models)
                    
                    # Get top models from patterns
                    top_pattern_models = [model for model, count in model_counts.most_common(3)]
                    
                    # Enhance model recommendations
                    if 'models' in enhanced_recommendations:
                        # Combine existing models with pattern-based models
                        existing_models = enhanced_recommendations['models']
                        enhanced_models = list(set(existing_models + top_pattern_models))
                        enhanced_recommendations['models'] = enhanced_models[:5]  # Keep top 5
                    
                    logger.info(f"🧠 Enhanced models with patterns: {top_pattern_models}")
                
                # Add pattern-based confidence
                enhanced_recommendations['pattern_confidence'] = len(similar_patterns) / 10.0  # Normalize to 0-1
                
            else:
                # No similar patterns found
                enhanced_recommendations = recommendations.copy()
                enhanced_recommendations['pattern_confidence'] = 0.0
            
            return enhanced_recommendations
            
        except Exception as e:
            logger.warning(f"⚠️ Meta-learning enhancement failed: {e}")
            return recommendations
    
    def _learn_from_selection(self, dataset_profile: Dict[str, Any], 
                             recommendations: Dict[str, Any]):
        """Learn from strategy selection for future improvements"""
        try:
            # Store pattern in pattern learner
            pattern_data = {
                'dataset_profile': dataset_profile,
                'strategy': recommendations,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            self.pattern_learner.add_pattern(pattern_data)
            
            # Update knowledge base
            if 'primary_strategy' in recommendations.get('metadata', {}):
                strategy_name = recommendations['metadata']['primary_strategy']
                self.knowledge_base.update_strategy_performance(
                    strategy_name, dataset_profile, 0.0  # Will be updated with actual performance
                )
            
            logger.debug("🧠 Learned from strategy selection")
            
        except Exception as e:
            logger.warning(f"⚠️ Learning from selection failed: {e}")
    
    def load_experiments(self):
        """Load experiments from file"""
        try:
            with open("knowledge_base.json", "r") as f:
                self.experiments = json.load(f)
            logger.info(f"📚 Loaded {len(self.experiments)} past experiments")
        except FileNotFoundError:
            logger.info("📚 No existing knowledge base found, starting fresh")
            self.experiments = []
        except Exception as e:
            logger.warning(f"Failed to load knowledge base: {e}")
            self.experiments = []
