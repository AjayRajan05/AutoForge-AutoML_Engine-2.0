"""
🎯 Actionable Insights Generator
Generate actionable recommendations from AutoML analysis
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ActionableInsight:
    """Single actionable insight"""
    category: str
    priority: str  # high, medium, low
    issue: str
    recommendation: str
    expected_impact: str
    implementation_effort: str  # low, medium, high
    confidence: float


class ActionableInsights:
    """
    Actionable Insights Generator
    
    Generates actionable recommendations from AutoML analysis including:
    - Data quality improvements
    - Feature engineering suggestions
    - Model optimization recommendations
    - Deployment considerations
    """
    
    def __init__(self):
        self.insights_history = []
        self.insight_templates = self._load_insight_templates()
        
    def generate_insights(self, X: pd.DataFrame, y: pd.Series, 
                         model_results: Dict[str, Any],
                         dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive actionable insights
        
        Args:
            X: Feature data
            y: Target data
            model_results: Model training results
            dataset_profile: Dataset analysis results
            
        Returns:
            Actionable insights and recommendations
        """
        logger.info("🎯 Generating actionable insights")
        
        insights = []
        
        # Data quality insights
        data_insights = self._analyze_data_quality(X, y, dataset_profile)
        insights.extend(data_insights)
        
        # Feature insights
        feature_insights = self._analyze_features(X, y, model_results, dataset_profile)
        insights.extend(feature_insights)
        
        # Model insights
        model_insights = self._analyze_model_performance(model_results, dataset_profile)
        insights.extend(model_insights)
        
        # Deployment insights
        deployment_insights = self._analyze_deployment_readiness(model_results, dataset_profile)
        insights.extend(deployment_insights)
        
        # Organize insights by priority
        priority_groups = self._organize_by_priority(insights)
        
        # Generate summary
        summary = self._generate_insights_summary(insights)
        
        insights_summary = {
            "timestamp": time.time(),
            "total_insights": len(insights),
            "insights": insights,
            "priority_groups": priority_groups,
            "summary": summary,
            "action_plan": self._generate_action_plan(priority_groups)
        }
        
        self.insights_history.append(insights_summary)
        
        logger.info(f"✅ Generated {len(insights)} actionable insights")
        return insights_summary
    
    def _analyze_data_quality(self, X: pd.DataFrame, y: pd.Series,
                            dataset_profile: Dict[str, Any]) -> List[ActionableInsight]:
        """Analyze data quality and generate insights"""
        insights = []
        
        # Missing values analysis
        quality_profile = dataset_profile.get("quality_profile", {})
        missing_ratio = quality_profile.get("missing_ratio", 0)
        
        if missing_ratio > 0.2:
            insights.append(ActionableInsight(
                category="data_quality",
                priority="high",
                issue=f"High missing values: {missing_ratio:.1%}",
                recommendation="Implement data collection improvements or advanced imputation",
                expected_impact="Improved model accuracy by 5-15%",
                implementation_effort="medium",
                confidence=0.9
            ))
        elif missing_ratio > 0.05:
            insights.append(ActionableInsight(
                category="data_quality",
                priority="medium",
                issue=f"Moderate missing values: {missing_ratio:.1%}",
                recommendation="Review data collection process and consider imputation strategies",
                expected_impact="Improved model reliability by 3-8%",
                implementation_effort="low",
                confidence=0.8
            ))
        
        # Duplicate data analysis
        duplicate_ratio = quality_profile.get("duplicate_ratio", 0)
        if duplicate_ratio > 0.05:
            insights.append(ActionableInsight(
                category="data_quality",
                priority="medium",
                issue=f"Duplicate records: {duplicate_ratio:.1%}",
                recommendation="Remove duplicate records and investigate data collection process",
                expected_impact="Improved model generalization by 2-5%",
                implementation_effort="low",
                confidence=0.85
            ))
        
        # Target variable analysis
        target_profile = dataset_profile.get("target_profile", {})
        if target_profile.get("type") == "classification":
            class_balance = target_profile.get("class_balance", 1.0)
            if class_balance < 0.7:  # Imbalanced
                insights.append(ActionableInsight(
                    category="data_quality",
                    priority="high",
                    issue=f"Imbalanced classes: ratio {class_balance:.2f}",
                    recommendation="Use class balancing techniques or collect more minority class data",
                    expected_impact="Improved minority class prediction by 10-30%",
                    implementation_effort="medium",
                    confidence=0.9
                ))
        
        # Outlier analysis
        if self._has_significant_outliers(X):
            insights.append(ActionableInsight(
                category="data_quality",
                priority="medium",
                issue="Significant outliers detected in numeric features",
                recommendation="Implement outlier detection and handling strategies",
                expected_impact="Improved model stability by 5-10%",
                implementation_effort="low",
                confidence=0.8
            ))
        
        return insights
    
    def _analyze_features(self, X: pd.DataFrame, y: pd.Series,
                        model_results: Dict[str, Any],
                        dataset_profile: Dict[str, Any]) -> List[ActionableInsight]:
        """Analyze features and generate insights"""
        insights = []
        
        # Feature correlation analysis
        correlation_profile = dataset_profile.get("correlation_profile", {})
        if correlation_profile.get("has_high_correlations", False):
            high_corr_pairs = correlation_profile.get("high_correlation_pairs", [])
            insights.append(ActionableInsight(
                category="feature_engineering",
                priority="medium",
                issue=f"Found {len(high_corr_pairs)} highly correlated feature pairs",
                recommendation="Apply feature selection or dimensionality reduction",
                expected_impact="Faster training and better generalization",
                implementation_effort="low",
                confidence=0.85
            ))
        
        # Feature type analysis
        type_profile = dataset_profile.get("type_profile", {})
        mixed_type_cols = type_profile.get("mixed_type_columns", [])
        
        if mixed_type_cols:
            insights.append(ActionableInsight(
                category="feature_engineering",
                priority="medium",
                issue=f"Found {len(mixed_type_cols)} columns with mixed data types",
                recommendation="Standardize data types and consider type-specific preprocessing",
                expected_impact="Improved preprocessing consistency and model performance",
                implementation_effort="medium",
                confidence=0.8
            ))
        
        # High cardinality categorical features
        categorical_cols = type_profile.get("categorical_columns", [])
        high_cardinality = []
        
        for col in categorical_cols:
            if col in X.columns:
                unique_ratio = X[col].nunique() / len(X)
                if unique_ratio > 0.5:  # More than 50% unique values
                    high_cardinality.append(col)
        
        if high_cardinality:
            insights.append(ActionableInsight(
                category="feature_engineering",
                priority="high",
                issue=f"High cardinality categorical features: {high_cardinality}",
                recommendation="Apply target encoding or feature grouping for high cardinality features",
                expected_impact="Improved model performance and reduced memory usage",
                implementation_effort="high",
                confidence=0.9
            ))
        
        # Feature importance analysis
        if "feature_importance" in model_results:
            feature_importance = model_results["feature_importance"]
            low_importance_features = [
                f for f, imp in feature_importance.items() 
                if imp < 0.01  # Less than 1% importance
            ]
            
            if len(low_importance_features) > 5:
                insights.append(ActionableInsight(
                    category="feature_engineering",
                    priority="low",
                    issue=f"Found {len(low_importance_features)} low-importance features",
                    recommendation="Consider removing low-importance features to simplify model",
                    expected_impact="Faster training with minimal accuracy loss",
                    implementation_effort="low",
                    confidence=0.7
                ))
        
        return insights
    
    def _analyze_model_performance(self, model_results: Dict[str, Any],
                                  dataset_profile: Dict[str, Any]) -> List[ActionableInsight]:
        """Analyze model performance and generate insights"""
        insights = []
        
        # Performance analysis
        best_score = model_results.get("best_score", 0)
        models_tried = model_results.get("models_tried", [])
        
        if best_score < 0.7:
            insights.append(ActionableInsight(
                category="model_optimization",
                priority="high",
                issue=f"Low model performance: {best_score:.3f}",
                recommendation="Try different models, feature engineering, or hyperparameter tuning",
                expected_impact="Significant improvement in model accuracy",
                implementation_effort="high",
                confidence=0.9
            ))
        elif best_score < 0.85:
            insights.append(ActionableInsight(
                category="model_optimization",
                priority="medium",
                issue=f"Moderate model performance: {best_score:.3f}",
                recommendation="Consider ensemble methods or advanced feature engineering",
                expected_impact="Improved model accuracy by 3-8%",
                implementation_effort="medium",
                confidence=0.8
            ))
        
        # Model diversity analysis
        if len(models_tried) < 3:
            insights.append(ActionableInsight(
                category="model_optimization",
                priority="medium",
                issue=f"Limited model diversity: only {len(models_tried)} models tried",
                recommendation="Try more diverse model types for better performance",
                expected_impact="Potential for better model selection",
                implementation_effort="low",
                confidence=0.7
            ))
        
        # Training time analysis
        training_time = model_results.get("training_time", 0)
        if training_time > 300:  # More than 5 minutes
            insights.append(ActionableInsight(
                category="model_optimization",
                priority="low",
                issue=f"Long training time: {training_time:.1f} seconds",
                recommendation="Consider model simplification or feature reduction for faster training",
                expected_impact="Reduced training time by 30-50%",
                implementation_effort="medium",
                confidence=0.8
            ))
        
        return insights
    
    def _analyze_deployment_readiness(self, model_results: Dict[str, Any],
                                    dataset_profile: Dict[str, Any]) -> List[ActionableInsight]:
        """Analyze deployment readiness and generate insights"""
        insights = []
        
        # Model complexity analysis
        best_model = model_results.get("best_model", "unknown")
        
        if "neural" in best_model.lower() or "deep" in best_model.lower():
            insights.append(ActionableInsight(
                category="deployment",
                priority="medium",
                issue="Complex neural network model selected",
                recommendation="Consider model compression or simpler alternatives for deployment",
                expected_impact="Faster inference and reduced resource requirements",
                implementation_effort="high",
                confidence=0.8
            ))
        
        # Data size considerations
        size_profile = dataset_profile.get("size_profile", {})
        n_samples = size_profile.get("n_samples", 0)
        
        if n_samples < 1000:
            insights.append(ActionableInsight(
                category="deployment",
                priority="medium",
                issue="Small training dataset",
                recommendation="Collect more training data or use data augmentation techniques",
                expected_impact="Improved model generalization in production",
                implementation_effort="high",
                confidence=0.9
            ))
        
        # Feature stability considerations
        feature_types = dataset_profile.get("feature_types", {})
        datetime_features = feature_types.get("datetime_features", [])
        
        if datetime_features:
            insights.append(ActionableInsight(
                category="deployment",
                priority="medium",
                issue="Datetime features detected",
                recommendation="Ensure datetime feature consistency in production environment",
                expected_impact="Prevent model failures due to data format issues",
                implementation_effort="low",
                confidence=0.85
            ))
        
        # Model monitoring recommendation
        insights.append(ActionableInsight(
            category="deployment",
            priority="low",
            issue="Model performance monitoring needed",
            recommendation="Implement model performance monitoring and drift detection",
            expected_impact="Early detection of model degradation",
            implementation_effort="medium",
            confidence=0.9
        ))
        
        return insights
    
    def _organize_by_priority(self, insights: List[ActionableInsight]) -> Dict[str, List[ActionableInsight]]:
        """Organize insights by priority level"""
        priority_groups = {
            "high": [],
            "medium": [],
            "low": []
        }
        
        for insight in insights:
            priority_groups[insight.priority].append(insight)
        
        return priority_groups
    
    def _generate_insights_summary(self, insights: List[ActionableInsight]) -> Dict[str, Any]:
        """Generate summary of insights"""
        categories = {}
        priorities = {"high": 0, "medium": 0, "low": 0}
        efforts = {"low": 0, "medium": 0, "high": 0}
        
        for insight in insights:
            # Count categories
            if insight.category not in categories:
                categories[insight.category] = 0
            categories[insight.category] += 1
            
            # Count priorities
            priorities[insight.priority] += 1
            
            # Count efforts
            efforts[insight.implementation_effort] += 1
        
        return {
            "total_insights": len(insights),
            "categories": categories,
            "priorities": priorities,
            "implementation_efforts": efforts,
            "most_common_category": max(categories.items(), key=lambda x: x[1])[0] if categories else None,
            "high_priority_count": priorities["high"],
            "average_confidence": np.mean([insight.confidence for insight in insights])
        }
    
    def _generate_action_plan(self, priority_groups: Dict[str, List[ActionableInsight]]) -> Dict[str, Any]:
        """Generate prioritized action plan"""
        action_plan = {
            "immediate_actions": [],  # High priority, low effort
            "short_term_actions": [],  # High priority, medium effort
            "long_term_actions": [],   # High priority, high effort
            "optimization_actions": [], # Medium priority
            "maintenance_actions": []   # Low priority
        }
        
        # High priority actions
        for insight in priority_groups["high"]:
            if insight.implementation_effort == "low":
                action_plan["immediate_actions"].append(insight)
            elif insight.implementation_effort == "medium":
                action_plan["short_term_actions"].append(insight)
            else:
                action_plan["long_term_actions"].append(insight)
        
        # Medium priority actions
        action_plan["optimization_actions"] = priority_groups["medium"]
        
        # Low priority actions
        action_plan["maintenance_actions"] = priority_groups["low"]
        
        return action_plan
    
    def _has_significant_outliers(self, X: pd.DataFrame) -> bool:
        """Check if dataset has significant outliers"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
            outlier_ratio = outliers / len(X)
            
            if outlier_ratio > 0.05:  # More than 5% outliers
                return True
        
        return False
    
    def _load_insight_templates(self) -> Dict[str, Any]:
        """Load insight templates for generating recommendations"""
        return {
            "data_quality": {
                "missing_values": {
                    "high": "Implement data collection improvements",
                    "medium": "Review data collection process",
                    "low": "Monitor data quality metrics"
                },
                "duplicates": {
                    "high": "Investigate data collection pipeline",
                    "medium": "Remove duplicates and review process",
                    "low": "Implement duplicate detection"
                }
            },
            "feature_engineering": {
                "correlation": {
                    "high": "Apply dimensionality reduction",
                    "medium": "Use feature selection",
                    "low": "Monitor feature correlations"
                },
                "cardinality": {
                    "high": "Implement target encoding",
                    "medium": "Group rare categories",
                    "low": "Review categorical features"
                }
            },
            "model_optimization": {
                "performance": {
                    "high": "Try advanced modeling techniques",
                    "medium": "Use ensemble methods",
                    "low": "Fine-tune hyperparameters"
                },
                "diversity": {
                    "high": "Expand model search space",
                    "medium": "Try different model families",
                    "low": "Evaluate more models"
                }
            }
        }
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary of actionable insights"""
        if not self.insights_history:
            return "No insights available for summary"
        
        latest = self.insights_history[-1]
        summary = latest["summary"]
        action_plan = latest["action_plan"]
        
        summary_lines = []
        
        summary_lines.append("🎯 ACTIONABLE INSIGHTS EXECUTIVE SUMMARY")
        summary_lines.append("=" * 60)
        
        summary_lines.append(f"Total Insights Generated: {summary['total_insights']}")
        summary_lines.append(f"High Priority Issues: {summary['high_priority_count']}")
        summary_lines.append(f"Average Confidence: {summary['average_confidence']:.1%}")
        
        summary_lines.append("\n📊 INSIGHT BREAKDOWN")
        summary_lines.append("-" * 30)
        
        for category, count in summary["categories"].items():
            summary_lines.append(f"{category.title()}: {count}")
        
        summary_lines.append("\n🚀 IMMEDIATE ACTIONS (High Priority, Low Effort)")
        summary_lines.append("-" * 30)
        
        for action in action_plan["immediate_actions"][:3]:  # Top 3
            summary_lines.append(f"• {action.issue}")
            summary_lines.append(f"  → {action.recommendation}")
        
        summary_lines.append("\n⏱️ SHORT-TERM ACTIONS")
        summary_lines.append("-" * 30)
        
        for action in action_plan["short_term_actions"][:2]:  # Top 2
            summary_lines.append(f"• {action.issue}")
            summary_lines.append(f"  → {action.recommendation}")
        
        summary_lines.append("\n📈 RECOMMENDED NEXT STEPS")
        summary_lines.append("-" * 30)
        
        if summary["high_priority_count"] > 0:
            summary_lines.append("1. Address all high-priority issues immediately")
            summary_lines.append("2. Implement data quality improvements")
            summary_lines.append("3. Set up model performance monitoring")
        else:
            summary_lines.append("1. Focus on optimization opportunities")
            summary_lines.append("2. Consider feature engineering improvements")
            summary_lines.append("3. Plan for production deployment")
        
        summary_lines.append("=" * 60)
        
        return "\n".join(summary_lines)
    
    def get_insight_summary(self) -> Dict[str, Any]:
        """Get summary of all insights"""
        if not self.insights_history:
            return {"error": "No insights history available"}
        
        return {
            "total_insight_sessions": len(self.insights_history),
            "latest_insights": self.insights_history[-1],
            "all_categories": list(set(
                insight.category
                for session in self.insights_history
                for insight in session["insights"]
            )),
            "total_insights_generated": sum(
                len(session["insights"]) 
                for session in self.insights_history
            )
        }
    
    def reset(self):
        """Reset insights history"""
        self.insights_history.clear()
        logger.info("🔄 Actionable insights generator reset")
