"""
🧠 Intelligence Engine - Unified AI Analysis System
Combines dataset analysis, strategy selection, and data type detection
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Optional
from dataclasses import dataclass

from .dataset_analyzer import DatasetIntelligence
from .strategy_selector import StrategySelector
from .data_type_detector import DataTypeDetector

logger = logging.getLogger(__name__)


@dataclass
class DatasetProfile:
    """Unified dataset profile containing all analysis results"""
    size_profile: Dict[str, Any]
    quality_profile: Dict[str, Any]
    type_profile: Dict[str, Any]
    complexity_profile: Dict[str, Any]
    target_profile: Dict[str, Any]
    data_type: str
    recommendations: Dict[str, Any]
    confidence: float
    analysis_metadata: Dict[str, Any]


class IntelligenceEngine:
    """
    Unified Intelligence Engine
    
    Combines dataset analysis, strategy selection, and data type detection
    to provide comprehensive intelligence for AutoML decision making.
    """
    
    def __init__(self):
        """Initialize all intelligence components"""
        logger.info("🧠 Initializing Intelligence Engine...")
        
        # Core analyzers
        self.dataset_analyzer = DatasetIntelligence()
        self.strategy_selector = StrategySelector()
        self.data_type_detector = DataTypeDetector()
        
        # Cache for analysis results
        self.analysis_cache = {}
        self.profile_cache = {}
        
        logger.info("✅ Intelligence Engine initialized with all analyzers")
    
    def analyze(self, X: Union[pd.DataFrame, np.ndarray], 
               y: Union[pd.Series, np.ndarray],
               user_preference: str = "auto",
               max_time: Optional[float] = None,
               max_trials: Optional[int] = None) -> DatasetProfile:
        """
        Comprehensive analysis of dataset and strategy selection
        
        Args:
            X: Feature data
            y: Target data
            user_preference: User preference ("auto", "fast", "accurate", "robust")
            max_time: Maximum time constraint
            max_trials: Maximum trials constraint
            
        Returns:
            Complete dataset profile with strategy recommendations
            
        Raises:
            ValueError: If input validation fails
            TypeError: If input types are incorrect
        """
        # Input validation
        if X is None:
            raise ValueError("X cannot be None")
        
        if y is None:
            raise ValueError("y cannot be None")
        
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be pandas DataFrame or numpy array")
        
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be pandas Series or numpy array")
        
        if user_preference not in ["auto", "fast", "accurate", "robust"]:
            raise ValueError("user_preference must be one of: 'auto', 'fast', 'accurate', 'robust'")
        
        if max_time is not None and (not isinstance(max_time, (int, float)) or max_time <= 0):
            raise ValueError("max_time must be a positive number or None")
        
        if max_trials is not None and (not isinstance(max_trials, int) or max_trials <= 0):
            raise ValueError("max_trials must be a positive integer or None")
        
        try:
            logger.info("🔍 Starting comprehensive dataset analysis...")
            
            # Convert to DataFrame for consistency
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            
            if not isinstance(y, (pd.Series, np.ndarray)):
                y = pd.Series(y)
            
            # Generate cache key
            cache_key = self._generate_cache_key(X, y, user_preference, max_time, max_trials)
            
            # Check cache
            if cache_key in self.profile_cache:
                logger.info("📋 Using cached analysis results")
                return self.profile_cache[cache_key]
            
            # Step 1: Dataset Analysis
            logger.info("📊 Performing dataset analysis...")
            dataset_analysis = self.dataset_analyzer.analyze(X, y)
            
            # Step 2: Data Type Detection
            logger.info("🔬 Detecting data types...")
            data_type_analysis = self.data_type_detector.detect_data_type(X, y)
            
            # Step 3: Strategy Selection
            logger.info("🎯 Selecting optimal strategy...")
            strategy = self.strategy_selector.select_strategy(
                dataset_analysis, user_preference, max_time, max_trials
            )
            
            # Step 4: Create unified profile
            profile = self._create_unified_profile(
                dataset_analysis, data_type_analysis, strategy
            )
            
            # Cache results
            self.profile_cache[cache_key] = profile
            
            logger.info(f"✅ Analysis complete. Data type: {profile.data_type}, "
                       f"Strategy: {profile.recommendations.get('primary_strategy', 'unknown')}")
            
            return profile
            
        except Exception as e:
            logger.error(f"❌ Intelligence analysis failed: {e}")
            return self._get_fallback_profile(X, y)
    
    def get_quick_analysis(self, X: Union[pd.DataFrame, np.ndarray], 
                          y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Quick analysis for basic insights
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Basic analysis results
        """
        try:
            # Convert to DataFrame
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            
            # Quick stats
            quick_stats = {
                "shape": X.shape,
                "target_type": "classification" if len(y.unique()) < 20 else "regression",
                "n_classes": len(y.unique()) if len(y.unique()) < 20 else 1,
                "missing_ratio": X.isnull().sum().sum() / (X.shape[0] * X.shape[1]),
                "numeric_features": X.select_dtypes(include=[np.number]).shape[1],
                "categorical_features": X.select_dtypes(include=['object', 'category']).shape[1],
                "memory_mb": X.memory_usage(deep=True).sum() / (1024 * 1024)
            }
            
            # Quick recommendations
            recommendations = []
            
            if quick_stats["missing_ratio"] > 0.2:
                recommendations.append("Consider robust imputation")
            
            if quick_stats["categorical_features"] > 0:
                recommendations.append("Apply categorical encoding")
            
            if quick_stats["shape"][0] > 100000:
                recommendations.append("Consider sampling for large datasets")
            
            quick_stats["recommendations"] = recommendations
            
            return quick_stats
            
        except Exception as e:
            logger.error(f"❌ Quick analysis failed: {e}")
            return {"error": "Quick analysis failed"}
    
    def learn_from_execution(self, profile: DatasetProfile, 
                           performance_score: float):
        """
        Learn from execution results to improve future selections
        
        Args:
            profile: Dataset profile that was used
            performance_score: Performance achieved (0-1)
        """
        try:
            # Extract dataset profile dict for learning
            dataset_profile = {
                "size_profile": profile.size_profile,
                "quality_profile": profile.quality_profile,
                "type_profile": profile.type_profile,
                "complexity_profile": profile.complexity_profile,
                "target_profile": profile.target_profile
            }
            
            # Extract strategy for learning
            strategy = {
                "preprocessing": profile.recommendations.get("preprocessing", []),
                "feature_engineering": profile.recommendations.get("feature_engineering", []),
                "models": profile.recommendations.get("models", []),
                "optimization": profile.recommendations.get("optimization", {}),
                "validation": profile.recommendations.get("validation", {})
            }
            
            # Learn from results
            self.strategy_selector.learn_from_result(dataset_profile, strategy, performance_score)
            
            logger.info(f"🧠 Learned from execution: performance={performance_score:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Learning from execution failed: {e}")
    
    def get_analysis_summary(self, profile: DatasetProfile) -> str:
        """
        Get human-readable summary of analysis results
        
        Args:
            profile: Dataset profile
            
        Returns:
            Human-readable summary
        """
        try:
            summary_lines = []
            
            # Basic info
            summary_lines.append(f"📊 Dataset: {profile.size_profile['n_samples']:,} samples, "
                               f"{profile.size_profile['n_features']} features")
            summary_lines.append(f"🎯 Task: {profile.complexity_profile['task_type']}")
            
            if profile.complexity_profile['task_type'] == 'classification':
                summary_lines.append(f"📈 Classes: {profile.complexity_profile['n_classes']}")
                if profile.complexity_profile['is_imbalanced']:
                    summary_lines.append("⚠️  Imbalanced classes detected")
            
            # Data quality
            quality_desc = profile.quality_profile['description']
            summary_lines.append(f"✨ Data Quality: {quality_desc}")
            
            # Data type
            summary_lines.append(f"🔬 Primary Data Type: {profile.data_type}")
            
            # Strategy
            strategy = profile.recommendations.get('primary_strategy', 'unknown')
            confidence = profile.confidence
            summary_lines.append(f"🎯 Recommended Strategy: {strategy} (confidence: {confidence:.1%})")
            
            # Key recommendations
            preprocessing = profile.recommendations.get('preprocessing', [])[:3]
            if preprocessing:
                summary_lines.append(f"🔧 Key Preprocessing: {', '.join(preprocessing)}")
            
            models = profile.recommendations.get('models', [])[:3]
            if models:
                summary_lines.append(f"🤖 Recommended Models: {', '.join(models)}")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate summary: {e}")
            return "Summary generation failed"
    
    def _create_unified_profile(self, dataset_analysis: Dict[str, Any],
                              data_type_analysis: Dict[str, Any],
                              strategy: Dict[str, Any]) -> DatasetProfile:
        """Create unified dataset profile from all analyses"""
        
        # Extract profiles from analysis
        size_profile = dataset_analysis.get("size_profile", {})
        quality_profile = dataset_analysis.get("quality_profile", {})
        type_profile = dataset_analysis.get("type_profile", {})
        complexity_profile = dataset_analysis.get("complexity_profile", {})
        target_profile = dataset_analysis.get("target_profile", {})
        
        # Get primary data type
        primary_data_type = data_type_analysis.get("primary_type", "tabular")
        
        # Combine recommendations
        combined_recommendations = {
            "preprocessing": dataset_analysis.get("recommendations", {}).get("preprocessing", []),
            "feature_engineering": dataset_analysis.get("recommendations", {}).get("feature_engineering", []),
            "model_selection": dataset_analysis.get("recommendations", {}).get("model_selection", []),
            "optimization": dataset_analysis.get("recommendations", {}).get("optimization", []),
            # Add strategy-specific recommendations
            "strategy_preprocessing": strategy.get("preprocessing", []),
            "strategy_feature_engineering": strategy.get("feature_engineering", []),
            "models": strategy.get("models", []),
            "optimization": strategy.get("optimization", {}),
            "validation": strategy.get("validation", {}),
            "primary_strategy": strategy.get("metadata", {}).get("primary_strategy", "unknown")
        }
        
        # Calculate overall confidence
        strategy_confidence = strategy.get("metadata", {}).get("confidence", 0.5)
        data_type_confidence = data_type_analysis.get("detection_results", {}).get(primary_data_type, {}).get("confidence", 0.5)
        overall_confidence = (strategy_confidence + data_type_confidence) / 2
        
        # Analysis metadata
        analysis_metadata = {
            "analysis_timestamp": strategy.get("metadata", {}).get("selection_time"),
            "data_type_confidence": data_type_confidence,
            "strategy_confidence": strategy_confidence,
            "data_type_recommendations": data_type_analysis.get("recommendations", []),
            "processing_pipeline": data_type_analysis.get("processing_pipeline", {})
        }
        
        return DatasetProfile(
            size_profile=size_profile,
            quality_profile=quality_profile,
            type_profile=type_profile,
            complexity_profile=complexity_profile,
            target_profile=target_profile,
            data_type=primary_data_type,
            recommendations=combined_recommendations,
            confidence=overall_confidence,
            analysis_metadata=analysis_metadata
        )
    
    def _generate_cache_key(self, X: Union[pd.DataFrame, np.ndarray], 
                           y: Union[pd.Series, np.ndarray], user_preference: str, max_time: Optional[float],
                           max_trials: Optional[int]) -> str:
        """Generate cache key for analysis results"""
        # Fix dtype sum issue - convert to string representation
        dtypes_str = str(X.dtypes.tolist()) if hasattr(X, 'dtypes') else str(X.dtype)
        return f"{X.shape}_{dtypes_str}_{len(y.unique())}_{user_preference}_{max_time}_{max_trials}"
    
    def _get_fallback_profile(self, X: Union[pd.DataFrame, np.ndarray], 
                            y: Union[pd.Series, np.ndarray]) -> DatasetProfile:
        """Get fallback profile when analysis fails"""
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Basic fallback profiles
        size_profile = {"category": "unknown", "n_samples": len(X), "n_features": X.shape[1]}
        quality_profile = {"category": "unknown", "missing_ratio": 0}
        type_profile = {"category": "unknown", "has_categorical": False}
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if len(y.unique()) < 20:
            inferred_task = "classification"
            n_classes = len(y.unique())
        elif pd.api.types.is_numeric_dtype(y):
            inferred_task = "regression"
            n_classes = 1
        else:
            inferred_task = "classification"
            n_classes = len(y.unique())
        complexity_profile = {"task_type": inferred_task, "n_classes": n_classes}
        target_profile = {"type": inferred_task}
        
        # Fallback recommendations
        recommendations = {
            "preprocessing": ["simple_imputation"],
            "feature_engineering": [],
            "models": ["random_forest"],
            "optimization": {"max_trials": 20},
            "validation": {"cv_folds": 3},
            "primary_strategy": "fallback"
        }
        
        return DatasetProfile(
            size_profile=size_profile,
            quality_profile=quality_profile,
            type_profile=type_profile,
            complexity_profile=complexity_profile,
            target_profile=target_profile,
            data_type="tabular",
            recommendations=recommendations,
            confidence=0.3,
            analysis_metadata={"fallback": True, "error": "Analysis failed"}
        )
    
    def clear_cache(self):
        """Clear all analysis caches"""
        self.analysis_cache.clear()
        self.profile_cache.clear()
        self.dataset_analyzer.analysis_cache.clear()
        logger.info("🧹 Cleared all analysis caches")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "analysis_cache_size": len(self.analysis_cache),
            "profile_cache_size": len(self.profile_cache),
            "dataset_analyzer_cache_size": len(self.dataset_analyzer.analysis_cache),
            "strategy_history_size": len(self.strategy_selector.strategy_history),
            "knowledge_base_experiments": len(self.strategy_selector.knowledge_base.experiments)
        }
