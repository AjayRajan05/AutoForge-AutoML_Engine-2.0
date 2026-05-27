"""
💡 Decision Explainer
Explainable AI for AutoML decision-making process
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DecisionExplanation:
    """Explanation for a specific decision"""
    factor: str
    observation: str
    decision: str
    reasoning: str
    confidence: float
    impact: str


class DecisionExplainer:
    """
    Decision Explainer
    
    Provides explanations for AutoML decisions including:
    - Strategy selection reasoning
    - Model choice explanations
    - Preprocessing decisions
    - Feature engineering choices
    """
    
    def __init__(self):
        self.explanation_history = []
        self.decision_rules = self._load_decision_rules()
        
    def explain_strategy_selection(self, dataset_profile: Dict[str, Any], 
                                 selected_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain why a particular strategy was selected
        
        Args:
            dataset_profile: Dataset characteristics
            selected_strategy: The strategy that was selected
            
        Returns:
            Explanation of strategy selection
        """
        logger.info("💡 Explaining strategy selection")
        
        explanations = []
        
        # Size-based explanations
        size_profile = dataset_profile.get("size_profile", {})
        size_category = size_profile.get("category", "unknown")
        
        if size_category == "large":
            explanations.append(DecisionExplanation(
                factor="dataset_size",
                observation=f"Large dataset detected ({size_profile.get('n_samples', 0):,} samples)",
                decision="Used sampling and fast models",
                reasoning="Large datasets benefit from sampling to reduce training time and fast models for efficiency",
                confidence=0.9,
                impact="Reduces training time by ~70%"
            ))
        elif size_category == "small":
            explanations.append(DecisionExplanation(
                factor="dataset_size",
                observation=f"Small dataset detected ({size_profile.get('n_samples', 0)} samples)",
                decision="Used cross-validation with more folds",
                reasoning="Small datasets benefit from more robust validation to prevent overfitting",
                confidence=0.85,
                impact="Improves model reliability"
            ))
        
        # Quality-based explanations
        quality_profile = dataset_profile.get("quality_profile", {})
        quality_category = quality_profile.get("category", "good")
        missing_ratio = quality_profile.get("missing_ratio", 0)
        
        if quality_category == "poor":
            explanations.append(DecisionExplanation(
                factor="data_quality",
                observation=f"High missing values detected ({missing_ratio:.1%})",
                decision="Applied robust preprocessing",
                reasoning="High missing values require robust imputation methods to maintain data integrity",
                confidence=0.95,
                impact="Prevents model failure due to missing data"
            ))
        
        # Type-based explanations
        type_profile = dataset_profile.get("type_profile", {})
        has_categorical = type_profile.get("has_categorical", False)
        n_categorical = len(type_profile.get("categorical_columns", []))
        
        if has_categorical:
            explanations.append(DecisionExplanation(
                factor="data_types",
                observation=f"Found {n_categorical} categorical features",
                decision="Applied categorical encoding",
                reasoning="Machine learning models require numeric input, categorical features need encoding",
                confidence=0.9,
                impact="Enables models to process categorical information"
            ))
        
        # Complexity-based explanations
        complexity_profile = dataset_profile.get("complexity_profile", {})
        is_imbalanced = complexity_profile.get("is_imbalanced", False)
        
        if is_imbalanced:
            explanations.append(DecisionExplanation(
                factor="class_balance",
                observation="Imbalanced dataset detected",
                decision="Applied class balancing techniques",
                reasoning="Imbalanced datasets can bias models toward majority class",
                confidence=0.85,
                impact="Improves minority class prediction"
            ))
        
        # Model-specific explanations
        models = selected_strategy.get("models", [])
        if "xgboost" in models:
            explanations.append(DecisionExplanation(
                factor="model_selection",
                observation="XGBoost selected as primary model",
                decision="Gradient boosting approach",
                reasoning="XGBoost typically provides best performance for tabular data with good speed/accuracy trade-off",
                confidence=0.8,
                impact="Optimizes for predictive accuracy"
            ))
        
        if "random_forest" in models:
            explanations.append(DecisionExplanation(
                factor="model_selection",
                observation="Random Forest selected",
                decision="Ensemble tree-based approach",
                reasoning="Random Forest is robust to outliers and handles mixed data types well",
                confidence=0.75,
                impact="Provides stable and reliable predictions"
            ))
        
        # Store explanation
        explanation_summary = {
            "timestamp": time.time(),
            "dataset_profile": dataset_profile,
            "selected_strategy": selected_strategy,
            "explanations": explanations,
            "summary": self._generate_explanation_summary(explanations)
        }
        
        self.explanation_history.append(explanation_summary)
        
        return explanation_summary
    
    def explain_model_performance(self, model_name: str, performance: float,
                                dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain model performance in context
        
        Args:
            model_name: Name of the model
            performance: Performance score
            dataset_profile: Dataset characteristics
            
        Returns:
            Performance explanation
        """
        logger.info(f"💡 Explaining {model_name} performance")
        
        # Benchmark performance
        benchmark_score = self._get_benchmark_score(model_name, dataset_profile)
        performance_delta = performance - benchmark_score
        
        explanations = []
        
        if performance_delta > 0.1:
            explanations.append({
                "factor": "excellent_performance",
                "observation": f"Performance {performance:.3f} exceeds benchmark by {performance_delta:.3f}",
                "reasoning": "Model performed exceptionally well on this dataset",
                "assessment": "excellent"
            })
        elif performance_delta > 0:
            explanations.append({
                "factor": "good_performance",
                "observation": f"Performance {performance:.3f} exceeds benchmark by {performance_delta:.3f}",
                "reasoning": "Model performed better than expected",
                "assessment": "good"
            })
        elif performance_delta > -0.05:
            explanations.append({
                "factor": "expected_performance",
                "observation": f"Performance {performance:.3f} close to benchmark",
                "reasoning": "Model performed as expected for this dataset type",
                "assessment": "expected"
            })
        else:
            explanations.append({
                "factor": "poor_performance",
                "observation": f"Performance {performance:.3f} below benchmark by {abs(performance_delta):.3f}",
                "reasoning": "Model underperformed, possibly due to dataset characteristics",
                "assessment": "below_expected"
            })
        
        # Dataset-specific factors
        size_category = dataset_profile.get("size_profile", {}).get("category", "medium")
        if size_category == "small" and performance > 0.8:
            explanations.append({
                "factor": "small_dataset_advantage",
                "observation": "High performance on small dataset",
                "reasoning": "Small datasets often allow models to achieve high accuracy",
                "assessment": "advantageous"
            })
        
        return {
            "model_name": model_name,
            "performance": performance,
            "benchmark_score": benchmark_score,
            "performance_delta": performance_delta,
            "explanations": explanations,
            "overall_assessment": explanations[0]["assessment"] if explanations else "unknown"
        }
    
    def explain_preprocessing_decisions(self, preprocessing_steps: List[str],
                                      dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain preprocessing decisions
        
        Args:
            preprocessing_steps: List of preprocessing steps applied
            dataset_profile: Dataset characteristics
            
        Returns:
            Preprocessing explanations
        """
        logger.info("💡 Explaining preprocessing decisions")
        
        explanations = []
        
        for step in preprocessing_steps:
            if step == "missing_values":
                missing_ratio = dataset_profile.get("quality_profile", {}).get("missing_ratio", 0)
                explanations.append({
                    "step": "missing_value_handling",
                    "reason": f"Missing value ratio: {missing_ratio:.1%}",
                    "necessity": "required" if missing_ratio > 0 else "preventive",
                    "impact": "Ensures model can process all data"
                })
            
            elif step == "categorical_encoding":
                n_categorical = len(dataset_profile.get("type_profile", {}).get("categorical_columns", []))
                explanations.append({
                    "step": "categorical_encoding",
                    "reason": f"Found {n_categorical} categorical features",
                    "necessity": "required",
                    "impact": "Converts categorical data to numeric format"
                })
            
            elif step == "feature_scaling":
                explanations.append({
                    "step": "feature_scaling",
                    "reason": "Features have different scales",
                    "necessity": "recommended",
                    "impact": "Improves model convergence and performance"
                })
            
            elif step == "outlier_removal":
                explanations.append({
                    "step": "outlier_handling",
                    "reason": "Potential outliers detected in numeric features",
                    "necessity": "recommended",
                    "impact": "Reduces influence of extreme values"
                })
        
        return {
            "steps_applied": preprocessing_steps,
            "explanations": explanations,
            "necessity_summary": {
                "required": len([e for e in explanations if e["necessity"] == "required"]),
                "recommended": len([e for e in explanations if e["necessity"] == "recommended"]),
                "preventive": len([e for e in explanations if e["necessity"] == "preventive"])
            }
        }
    
    def explain_feature_importance(self, feature_importance: Dict[str, float],
                                dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain feature importance results
        
        Args:
            feature_importance: Dictionary of feature importance scores
            dataset_profile: Dataset characteristics
            
        Returns:
            Feature importance explanations
        """
        logger.info("💡 Explaining feature importance")
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        top_features = sorted_features[:10]
        total_importance = sum(feature_importance.values())
        
        explanations = []
        
        for feature, importance in top_features:
            relative_importance = importance / total_importance
            
            if relative_importance > 0.2:
                tier = "critical"
                explanation = "This feature dominates the model's predictions"
            elif relative_importance > 0.1:
                tier = "important"
                explanation = "This feature significantly influences predictions"
            elif relative_importance > 0.05:
                tier = "moderate"
                explanation = "This feature contributes to model decisions"
            else:
                tier = "minor"
                explanation = "This feature has minimal impact"
            
            explanations.append({
                "feature": feature,
                "importance": importance,
                "relative_importance": relative_importance,
                "tier": tier,
                "explanation": explanation
            })
        
        # Feature type analysis
        feature_types = dataset_profile.get("feature_types", {})
        numeric_features = feature_types.get("numeric_features", [])
        categorical_features = feature_types.get("categorical_features", [])
        
        type_analysis = {
            "top_numeric": [f for f, _ in top_features if f in numeric_features],
            "top_categorical": [f for f, _ in top_features if f in categorical_features]
        }
        
        return {
            "total_features": len(feature_importance),
            "top_features": explanations,
            "importance_distribution": {
                "critical": len([e for e in explanations if e["tier"] == "critical"]),
                "important": len([e for e in explanations if e["tier"] == "important"]),
                "moderate": len([e for e in explanations if e["tier"] == "moderate"]),
                "minor": len([e for e in explanations if e["tier"] == "minor"])
            },
            "type_analysis": type_analysis,
            "concentration": len(top_features) / len(feature_importance)
        }
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate comprehensive explanation report
        
        Returns:
            Human-readable explanation report
        """
        if not self.explanation_history:
            return "No explanation history available"
        
        latest = self.explanation_history[-1]
        
        report_lines = []
        
        report_lines.append("💡 AUTOML DECISION EXPLANATION REPORT")
        report_lines.append("=" * 60)
        
        # Strategy selection explanation
        report_lines.append("\n🎯 STRATEGY SELECTION")
        report_lines.append("-" * 30)
        
        for explanation in latest["explanations"]:
            report_lines.append(f"Factor: {explanation.factor}")
            report_lines.append(f"  Observation: {explanation.observation}")
            report_lines.append(f"  Decision: {explanation.decision}")
            report_lines.append(f"  Reasoning: {explanation.reasoning}")
            report_lines.append(f"  Impact: {explanation.impact}")
            report_lines.append(f"  Confidence: {explanation.confidence:.1%}")
            report_lines.append("")
        
        # Summary
        summary = latest["summary"]
        report_lines.append("📊 SUMMARY")
        report_lines.append("-" * 30)
        report_lines.append(f"Primary Factors: {', '.join(summary['primary_factors'])}")
        report_lines.append(f"Overall Confidence: {summary['overall_confidence']:.1%}")
        report_lines.append(f"Key Considerations: {len(summary['key_considerations'])}")
        
        for consideration in summary["key_considerations"]:
            report_lines.append(f"  • {consideration}")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def _generate_explanation_summary(self, explanations: List[DecisionExplanation]) -> Dict[str, Any]:
        """Generate summary from explanations"""
        primary_factors = [exp.factor for exp in explanations]
        key_considerations = [exp.reasoning for exp in explanations]
        overall_confidence = np.mean([exp.confidence for exp in explanations])
        
        return {
            "primary_factors": primary_factors,
            "key_considerations": key_considerations,
            "overall_confidence": overall_confidence,
            "total_explanations": len(explanations)
        }
    
    def _get_benchmark_score(self, model_name: str, dataset_profile: Dict[str, Any]) -> float:
        """Get benchmark score for model on similar dataset"""
        # Simplified benchmark scores
        benchmarks = {
            "random_forest": 0.85,
            "xgboost": 0.87,
            "logistic_regression": 0.78,
            "svm": 0.82,
            "linear_regression": 0.75,
            "svr": 0.77
        }
        
        base_score = benchmarks.get(model_name, 0.8)
        
        # Adjust based on dataset characteristics
        size_category = dataset_profile.get("size_profile", {}).get("category", "medium")
        quality_category = dataset_profile.get("quality_profile", {}).get("category", "good")
        
        if size_category == "small":
            base_score += 0.05  # Small datasets often get higher scores
        elif size_category == "large":
            base_score -= 0.03  # Large datasets might have lower scores
        
        if quality_category == "poor":
            base_score -= 0.1  # Poor quality reduces performance
        
        return max(0.5, min(0.95, base_score))  # Clamp between 0.5 and 0.95
    
    def _load_decision_rules(self) -> Dict[str, Any]:
        """Load decision rules for explanations"""
        return {
            "size_decisions": {
                "small": {"cv_folds": 5, "max_trials": 100},
                "medium": {"cv_folds": 5, "max_trials": 75},
                "large": {"cv_folds": 3, "max_trials": 50}
            },
            "quality_decisions": {
                "good": {"preprocessing": "minimal"},
                "moderate": {"preprocessing": "standard"},
                "poor": {"preprocessing": "robust"}
            },
            "model_preferences": {
                "classification": ["xgboost", "random_forest", "logistic_regression"],
                "regression": ["xgboost", "random_forest", "linear_regression"]
            }
        }
    
    def get_explanation_summary(self) -> Dict[str, Any]:
        """Get summary of all explanations"""
        if not self.explanation_history:
            return {"error": "No explanation history available"}
        
        return {
            "total_explanations": len(self.explanation_history),
            "latest_explanation": self.explanation_history[-1],
            "all_factors": list(set(
                factor for exp in self.explanation_history 
                for explanation in exp["explanations"]
                for factor in [explanation.factor]
            )),
            "average_confidence": np.mean([
                explanation.confidence
                for exp in self.explanation_history
                for explanation in exp["explanations"]
            ])
        }
    
    def reset(self):
        """Reset explanation history"""
        self.explanation_history.clear()
        logger.info("🔄 Decision explainer reset")
