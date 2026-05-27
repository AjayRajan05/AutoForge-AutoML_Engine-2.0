"""
Pattern Learning System
Learns from past experiments to improve future performance
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
from pathlib import Path
import datetime

logger = logging.getLogger(__name__)


class PatternLearner:
    """
    Learns patterns from past AutoML experiments to improve future performance
    """
    
    def __init__(self, patterns_file: str = "meta_learning/patterns.json"):
        """
        Initialize pattern learner
        
        Args:
            patterns_file: Path to store learned patterns
        """
        self.patterns_file = Path(patterns_file)
        self.patterns_file.parent.mkdir(exist_ok=True)
        
        # Load existing patterns
        self.patterns = self._load_patterns()
        
        # Initialize pattern storage
        self.model_performance_patterns = defaultdict(list)
        self.feature_engineering_patterns = defaultdict(list)
        self.dataset_characteristics = defaultdict(list)
        self.optimization_patterns = defaultdict(list)
        
        logger.info(f"Pattern learner initialized: {len(self.patterns)} existing patterns")
    
    def _load_patterns(self) -> Dict[str, Any]:
        """Load existing patterns from file"""
        try:
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "model_performance": {},
                    "feature_engineering": {},
                    "dataset_characteristics": {},
                    "optimization": {},
                    "last_updated": None
                }
        except Exception as e:
            logger.warning(f"Failed to load patterns: {e}")
            return {"model_performance": {}, "feature_engineering": {}, "dataset_characteristics": {}, "optimization": {}}
    
    def _save_patterns(self) -> None:
        """Save patterns to file"""
        try:
            patterns_data = {
                "model_performance": dict(self.model_performance_patterns),
                "feature_engineering": dict(self.feature_engineering_patterns),
                "dataset_characteristics": dict(self.dataset_characteristics),
                "optimization": dict(self.optimization_patterns),
                "last_updated": datetime.datetime.now().isoformat()
            }
            
            with open(self.patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2, default=str)
            
            logger.info("Patterns saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")
    
    def learn_from_experiment(self, 
                           experiment_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Learn patterns from a completed experiment
        
        Args:
            experiment_result: Results from an AutoML experiment
            
        Returns:
            Learned insights and recommendations
        """
        try:
            insights = {}
            
            # Extract experiment characteristics
            dataset_info = experiment_result.get("dataset_info", {})
            model_info = experiment_result.get("model_info", {})
            optimization_info = experiment_result.get("optimization_info", {})
            performance_metrics = experiment_result.get("performance_metrics", {})
            
            # Learn model performance patterns
            model_insights = self._learn_model_patterns(
                dataset_info, model_info, performance_metrics
            )
            insights["model_patterns"] = model_insights
            
            # Learn feature engineering patterns
            feature_insights = self._learn_feature_patterns(
                dataset_info, experiment_result.get("feature_engineering", {})
            )
            insights["feature_patterns"] = feature_insights
            
            # Learn optimization patterns
            optimization_insights = self._learn_optimization_patterns(
                dataset_info, optimization_info, performance_metrics
            )
            insights["optimization_patterns"] = optimization_insights
            
            # Update stored patterns
            self._update_patterns(experiment_result, insights)
            
            # Save updated patterns
            self._save_patterns()
            
            logger.info(f"Learned patterns from experiment: {len(insights)} insights")
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to learn from experiment: {e}")
            return {"error": str(e)}
    
    def _learn_model_patterns(self, 
                           dataset_info: Dict[str, Any],
                           model_info: Dict[str, Any],
                           performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Learn patterns about model performance"""
        patterns = {}
        
        # Dataset size patterns
        dataset_size = dataset_info.get("n_samples", 0)
        n_features = dataset_info.get("n_features", 0)
        model_name = model_info.get("name", "unknown")
        
        # Large dataset patterns
        if dataset_size > 100000:
            # XGBoost and LightGBM typically work best for large datasets
            if "xgboost" in model_name.lower() or "lightgbm" in model_name.lower():
                performance = performance_metrics.get("accuracy", 0) or performance_metrics.get("r2_score", 0)
                self.model_performance_patterns["large_dataset_best_models"].append({
                    "model": model_name,
                    "performance": performance,
                    "dataset_size": dataset_size,
                    "n_features": n_features
                })
                patterns["large_dataset_insight"] = f"{model_name} performs well on large datasets"
        
        # Small dataset patterns
        elif dataset_size < 1000:
            # Simpler models often work better for small datasets
            if "randomforest" in model_name.lower() or "logistic" in model_name.lower():
                performance = performance_metrics.get("accuracy", 0) or performance_metrics.get("r2_score", 0)
                self.model_performance_patterns["small_dataset_best_models"].append({
                    "model": model_name,
                    "performance": performance,
                    "dataset_size": dataset_size,
                    "n_features": n_features
                })
                patterns["small_dataset_insight"] = f"{model_name} performs well on small datasets"
        
        # High-dimensional patterns
        if n_features > 100:
            # Models with built-in feature selection work better
            if "randomforest" in model_name.lower() or "xgboost" in model_name.lower():
                performance = performance_metrics.get("accuracy", 0) or performance_metrics.get("r2_score", 0)
                self.model_performance_patterns["high_dimensional_best_models"].append({
                    "model": model_name,
                    "performance": performance,
                    "n_features": n_features
                })
                patterns["high_dimensional_insight"] = f"{model_name} handles high-dimensional data well"
        
        return patterns
    
    def _learn_feature_patterns(self, 
                            dataset_info: Dict[str, Any],
                            feature_engineering: Dict[str, Any]) -> Dict[str, Any]:
        """Learn patterns about feature engineering effectiveness"""
        patterns = {}
        
        dataset_size = dataset_info.get("n_samples", 0)
        feature_types = feature_engineering.get("feature_types", [])
        engineered_features = feature_engineering.get("engineered_features", 0)
        
        # TF-IDF patterns for small datasets
        if dataset_size < 1000 and "tfidf" in feature_types:
            # TF-IDF can hurt small datasets due to sparsity
            self.feature_engineering_patterns["tfidf_small_dataset"].append({
                "dataset_size": dataset_size,
                "impact": "negative",
                "reason": "sparsity in small datasets"
            })
            patterns["tfidf_insight"] = "TF-IDF features may hurt small datasets due to sparsity"
        
        # Polynomial features for large datasets
        if dataset_size > 50000 and "polynomial" in feature_types:
            # Polynomial features can cause memory issues in large datasets
            self.feature_engineering_patterns["polynomial_large_dataset"].append({
                "dataset_size": dataset_size,
                "impact": "negative",
                "reason": "memory and computational overhead"
            })
            patterns["polynomial_insight"] = "Polynomial features may cause memory issues in large datasets"
        
        # Time series patterns
        if "time_series" in feature_types:
            # Lag features generally help time series
            self.feature_engineering_patterns["time_series_lag_features"].append({
                "impact": "positive",
                "reason": "captures temporal dependencies"
            })
            patterns["time_series_insight"] = "Lag features are effective for time series data"
        
        return patterns
    
    def _learn_optimization_patterns(self, 
                                 dataset_info: Dict[str, Any],
                                 optimization_info: Dict[str, Any],
                                 performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Learn patterns about optimization effectiveness"""
        patterns = {}
        
        dataset_size = dataset_info.get("n_samples", 0)
        optimization_strategy = optimization_info.get("strategy", "unknown")
        n_trials = optimization_info.get("n_trials", 0)
        final_score = performance_metrics.get("accuracy", 0) or performance_metrics.get("r2_score", 0)
        
        # Trial budget patterns
        if n_trials > 100 and final_score < 0.8:
            # Too many trials with poor results might indicate wrong approach
            self.optimization_patterns["excessive_trials_poor_results"].append({
                "n_trials": n_trials,
                "final_score": final_score,
                "dataset_size": dataset_size
            })
            patterns["trial_insight"] = "High trial count with poor results suggests suboptimal approach"
        
        # Adaptive optimization patterns
        if optimization_strategy == "adaptive" and final_score > 0.85:
            # Adaptive optimization working well
            self.optimization_patterns["adaptive_optimization_success"].append({
                "strategy": "adaptive",
                "final_score": final_score,
                "dataset_size": dataset_size
            })
            patterns["adaptive_insight"] = "Adaptive optimization is effective for this dataset type"
        
        return patterns
    
    def _update_patterns(self, experiment_result: Dict[str, Any], insights: Dict[str, Any]) -> None:
        """Update internal pattern storage with new learnings"""
        # This would update the internal pattern storage
        # Already handled in the individual learning methods
        pass
    
    def get_recommendations(self, 
                         dataset_info: Dict[str, Any],
                         task_type: str = "classification") -> Dict[str, Any]:
        """
        Get recommendations based on learned patterns
        
        Args:
            dataset_info: Dataset characteristics
            task_type: Type of ML task
            
        Returns:
            Recommendations for models, features, and optimization
        """
        try:
            recommendations = {
                "models": [],
                "feature_engineering": [],
                "optimization": [],
                "confidence": {}
            }
            
            dataset_size = dataset_info.get("n_samples", 0)
            n_features = dataset_info.get("n_features", 0)
            data_type = dataset_info.get("data_type", "tabular")
            
            # Model recommendations
            model_recs = self._get_model_recommendations(
                dataset_size, n_features, data_type, task_type
            )
            recommendations["models"] = model_recs
            
            # Feature engineering recommendations
            feature_recs = self._get_feature_recommendations(
                dataset_size, n_features, data_type
            )
            recommendations["feature_engineering"] = feature_recs
            
            # Optimization recommendations
            opt_recs = self._get_optimization_recommendations(
                dataset_size, n_features
            )
            recommendations["optimization"] = opt_recs
            
            # Calculate confidence scores
            recommendations["confidence"] = self._calculate_confidence(
                dataset_size, n_features, data_type
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return {"error": str(e)}
    
    def _get_model_recommendations(self, 
                                dataset_size: int,
                                n_features: int,
                                data_type: str,
                                task_type: str) -> List[Dict[str, Any]]:
        """Get model recommendations based on patterns"""
        recommendations = []
        
        # Large dataset recommendations
        if dataset_size > 100000:
            large_models = self.model_performance_patterns.get("large_dataset_best_models", [])
            if large_models:
                # Find best performing models for large datasets
                model_performance = defaultdict(list)
                for entry in large_models:
                    model_performance[entry["model"]].append(entry["performance"])
                
                for model, performances in model_performance.items():
                    avg_performance = np.mean(performances)
                    if avg_performance > 0.8:  # Good performance threshold
                        recommendations.append({
                            "model": model,
                            "reason": "Performs well on large datasets",
                            "confidence": avg_performance,
                            "priority": "high"
                        })
        
        # Small dataset recommendations
        elif dataset_size < 1000:
            small_models = self.model_performance_patterns.get("small_dataset_best_models", [])
            if small_models:
                model_performance = defaultdict(list)
                for entry in small_models:
                    model_performance[entry["model"]].append(entry["performance"])
                
                for model, performances in model_performance.items():
                    avg_performance = np.mean(performances)
                    if avg_performance > 0.75:
                        recommendations.append({
                            "model": model,
                            "reason": "Performs well on small datasets",
                            "confidence": avg_performance,
                            "priority": "high"
                        })
        
        # High-dimensional recommendations
        if n_features > 100:
            high_dim_models = self.model_performance_patterns.get("high_dimensional_best_models", [])
            if high_dim_models:
                model_performance = defaultdict(list)
                for entry in high_dim_models:
                    model_performance[entry["model"]].append(entry["performance"])
                
                for model, performances in model_performance.items():
                    avg_performance = np.mean(performances)
                    if avg_performance > 0.8:
                        recommendations.append({
                            "model": model,
                            "reason": "Handles high-dimensional data well",
                            "confidence": avg_performance,
                            "priority": "medium"
                        })
        
        # Default recommendations if no patterns found
        if not recommendations:
            if task_type == "classification":
                recommendations.extend([
                    {"model": "RandomForest", "reason": "Robust baseline", "confidence": 0.7, "priority": "medium"},
                    {"model": "XGBoost", "reason": "Strong performance", "confidence": 0.8, "priority": "high"}
                ])
            else:
                recommendations.extend([
                    {"model": "RandomForest", "reason": "Robust baseline", "confidence": 0.7, "priority": "medium"},
                    {"model": "XGBoost", "reason": "Strong performance", "confidence": 0.8, "priority": "high"}
                ])
        
        return recommendations
    
    def _get_feature_recommendations(self, 
                                  dataset_size: int,
                                  n_features: int,
                                  data_type: str) -> List[Dict[str, Any]]:
        """Get feature engineering recommendations"""
        recommendations = []
        
        # TF-IDF warnings for small datasets
        if dataset_size < 1000:
            tfidf_patterns = self.feature_engineering_patterns.get("tfidf_small_dataset", [])
            if tfidf_patterns:
                negative_impacts = [p for p in tfidf_patterns if p.get("impact") == "negative"]
                if len(negative_impacts) > len(tfidf_patterns) * 0.7:  # 70% negative
                    recommendations.append({
                        "feature": "TF-IDF",
                        "action": "avoid",
                        "reason": "TF-IDF features hurt small datasets due to sparsity",
                        "confidence": 0.8,
                        "priority": "high"
                    })
        
        # Polynomial feature warnings for large datasets
        if dataset_size > 50000:
            poly_patterns = self.feature_engineering_patterns.get("polynomial_large_dataset", [])
            if poly_patterns:
                negative_impacts = [p for p in poly_patterns if p.get("impact") == "negative"]
                if len(negative_impacts) > len(poly_patterns) * 0.6:  # 60% negative
                    recommendations.append({
                        "feature": "Polynomial Features",
                        "action": "avoid",
                        "reason": "Polynomial features cause memory issues in large datasets",
                        "confidence": 0.7,
                        "priority": "medium"
                    })
        
        # Time series recommendations
        if data_type == "time_series":
            lag_patterns = self.feature_engineering_patterns.get("time_series_lag_features", [])
            if lag_patterns:
                positive_impacts = [p for p in lag_patterns if p.get("impact") == "positive"]
                if len(positive_impacts) > len(lag_patterns) * 0.6:  # 60% positive
                    recommendations.append({
                        "feature": "Lag Features",
                        "action": "include",
                        "reason": "Lag features are effective for time series data",
                        "confidence": 0.8,
                        "priority": "high"
                    })
        
        return recommendations
    
    def _get_optimization_recommendations(self, 
                                      dataset_size: int,
                                      n_features: int) -> List[Dict[str, Any]]:
        """Get optimization recommendations"""
        recommendations = []
        
        # Adaptive optimization recommendations
        adaptive_patterns = self.optimization_patterns.get("adaptive_optimization_success", [])
        if adaptive_patterns:
            avg_performance = np.mean([p["final_score"] for p in adaptive_patterns])
            if avg_performance > 0.85:
                recommendations.append({
                    "strategy": "Adaptive Optimization",
                    "reason": "Adaptive optimization has been successful",
                    "confidence": avg_performance,
                    "priority": "high"
                })
        
        # Trial budget recommendations
        excessive_patterns = self.optimization_patterns.get("excessive_trials_poor_results", [])
        if excessive_patterns:
            avg_trials = np.mean([p["n_trials"] for p in excessive_patterns])
            if avg_trials > 100:
                recommendations.append({
                    "strategy": "Limited Trial Budget",
                    "reason": "High trial counts with poor results suggest budget limits",
                    "confidence": 0.7,
                    "priority": "medium",
                    "suggested_trials": min(50, int(avg_trials * 0.5))
                })
        
        return recommendations
    
    def _calculate_confidence(self, 
                           dataset_size: int,
                           n_features: int,
                           data_type: str) -> Dict[str, float]:
        """Calculate confidence scores for recommendations"""
        confidence = {}
        
        # Confidence based on pattern count
        model_patterns = len(self.model_performance_patterns.get("large_dataset_best_models", [])) + \
                      len(self.model_performance_patterns.get("small_dataset_best_models", []))
        
        feature_patterns = len(self.feature_engineering_patterns.get("tfidf_small_dataset", [])) + \
                         len(self.feature_engineering_patterns.get("polynomial_large_dataset", []))
        
        optimization_patterns = len(self.optimization_patterns.get("adaptive_optimization_success", []))
        
        total_patterns = model_patterns + feature_patterns + optimization_patterns
        
        if total_patterns > 50:
            confidence["overall"] = 0.9
        elif total_patterns > 20:
            confidence["overall"] = 0.8
        elif total_patterns > 10:
            confidence["overall"] = 0.7
        else:
            confidence["overall"] = 0.5
        
        confidence["models"] = min(0.9, model_patterns / 20.0)
        confidence["features"] = min(0.9, feature_patterns / 15.0)
        confidence["optimization"] = min(0.9, optimization_patterns / 10.0)
        
        return confidence
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of learned patterns"""
        try:
            summary = {
                "total_patterns": len(self.patterns),
                "model_performance_patterns": len(self.patterns.get("model_performance", {})),
                "feature_engineering_patterns": len(self.patterns.get("feature_engineering", {})),
                "dataset_characteristics": len(self.patterns.get("dataset_characteristics", {})),
                "optimization_patterns": len(self.patterns.get("optimization", {}))
            }
            
            for category, patterns in self.optimization_patterns.items():
                summary["total_patterns"] += len(patterns)
                summary["optimization_insights"][category] = len(patterns)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get pattern summary: {e}")
            return {"error": str(e)}
    
    def find_similar_patterns(self, dataset_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find patterns similar to the current dataset profile"""
        try:
            similar_patterns = []
            
            # Extract key characteristics from dataset profile
            dataset_size = dataset_profile.get('size_profile', {}).get('n_samples', 1000)
            n_features = dataset_profile.get('size_profile', {}).get('n_features', 10)
            data_type = dataset_profile.get('type_profile', {}).get('primary_type', 'tabular')
            
            # Look for similar patterns in stored patterns
            all_patterns = self.patterns.get('model_performance', {})
            
            for pattern_name, pattern_data in all_patterns.items():
                if isinstance(pattern_data, list):
                    for pattern in pattern_data:
                        # Simple similarity check based on dataset characteristics
                        pattern_size = pattern.get('dataset_size', 1000)
                        pattern_features = pattern.get('n_features', 10)
                        
                        # Calculate similarity score
                        size_diff = abs(dataset_size - pattern_size) / max(dataset_size, pattern_size, 1)
                        feature_diff = abs(n_features - pattern_features) / max(n_features, pattern_features, 1)
                        
                        similarity_score = 1.0 - (size_diff + feature_diff) / 2
                        
                        if similarity_score > 0.3:  # Threshold for similarity
                            similar_patterns.append({
                                'pattern': pattern,
                                'similarity': similarity_score,
                                'best_model': pattern.get('best_model', 'random_forest')
                            })
            
            # Sort by similarity
            similar_patterns.sort(key=lambda x: x['similarity'], reverse=True)
            
            return similar_patterns[:5]  # Return top 5 similar patterns
            
        except Exception as e:
            logger.warning(f"Failed to find similar patterns: {e}")
            return []


# Convenience functions
def learn_from_experiment(experiment_result: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for learning from experiment"""
    learner = PatternLearner()
    return learner.learn_from_experiment(experiment_result)


def get_recommendations(dataset_info: Dict[str, Any], task_type: str = "classification") -> Dict[str, Any]:
    """Convenience function for getting recommendations"""
    learner = PatternLearner()
    return learner.get_recommendations(dataset_info, task_type)
