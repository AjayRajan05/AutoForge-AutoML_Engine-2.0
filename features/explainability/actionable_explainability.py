"""
Actionable Explainability System
Provides specific, actionable recommendations from model explanations
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)


class ActionableExplainability:
    """
    Provides actionable insights and recommendations from model explanations
    """
    
    def __init__(self):
        """Initialize actionable explainability system"""
        self.insights = {}
        logger.info("Actionable explainability system initialized")
    
    def generate_actionable_insights(self, 
                                   explanations: Dict[str, Any],
                                   X: pd.DataFrame,
                                   y: pd.Series = None) -> Dict[str, Any]:
        """
        Generate actionable insights from model explanations
        
        Args:
            explanations: Model explanations from explainability system
            X: Feature data
            y: Target data (optional)
            
        Returns:
            Actionable insights and recommendations
        """
        try:
            actionable_insights = {
                "feature_recommendations": [],
                "data_quality_issues": [],
                "model_risks": [],
                "optimization_suggestions": [],
                "business_insights": [],
                "action_priority": {}
            }
            
            # Get feature importance
            feature_importance = explanations.get("feature_importance", {})
            if not feature_importance:
                # Try to get from different methods
                for method in ["aggregated", "direct", "permutation"]:
                    if method in explanations.get("feature_importance", {}):
                        feature_importance = explanations["feature_importance"][method]
                        break
            
            # Handle case where feature_importance is still nested
            if isinstance(feature_importance, dict) and any(isinstance(v, dict) for v in feature_importance.values()):
                # Extract the first nested dictionary we find
                for key, value in feature_importance.items():
                    if isinstance(value, dict):
                        feature_importance = value
                        break
            
            if not feature_importance:
                logger.warning("No feature importance found in explanations")
                return {"error": "No feature importance available"}
            
            # Analyze feature importance patterns
            importance_insights = self._analyze_feature_importance_patterns(
                feature_importance, X
            )
            actionable_insights["feature_recommendations"] = importance_insights
            
            # Analyze data quality issues
            quality_insights = self._analyze_data_quality_issues(X, y, feature_importance)
            actionable_insights["data_quality_issues"] = quality_insights
            
            # Analyze model risks
            risk_insights = self._analyze_model_risks(feature_importance, X)
            actionable_insights["model_risks"] = risk_insights
            
            # Generate optimization suggestions
            optimization_insights = self._generate_optimization_suggestions(
                feature_importance, X, explanations
            )
            actionable_insights["optimization_suggestions"] = optimization_insights
            
            # Generate business insights
            business_insights = self._generate_business_insights(
                feature_importance, X, y
            )
            actionable_insights["business_insights"] = business_insights
            
            # Calculate action priorities
            actionable_insights["action_priority"] = self._calculate_action_priorities(
                actionable_insights
            )
            
            logger.info(f"Generated {len(actionable_insights['feature_recommendations'])} actionable insights")
            
            return actionable_insights
            
        except Exception as e:
            logger.error(f"Failed to generate actionable insights: {e}")
            return {"error": str(e)}
    
    def _analyze_feature_importance_patterns(self, 
                                        feature_importance: Dict[str, float],
                                        X: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze patterns in feature importance"""
        insights = []
        
        if not feature_importance:
            return insights
        
        # Convert to sorted list
        sorted_importance = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Check for dominant features
        if len(sorted_importance) > 1:
            top_importance = sorted_importance[0][1]
            second_importance = sorted_importance[1][1] if len(sorted_importance) > 1 else 0
            
            if top_importance > second_importance * 3:  # 3x more important
                insights.append({
                    "type": "dominant_feature",
                    "feature": sorted_importance[0][0],
                    "importance": top_importance,
                    "second_importance": second_importance,
                    "ratio": top_importance / second_importance,
                    "insight": f"Feature '{sorted_importance[0][0]}' dominates model decisions",
                    "recommendation": "Investigate potential overfitting to this feature",
                    "action": "validate_feature",
                    "priority": "high"
                })
        
        # Check for noisy features (very low importance)
        total_importance = sum(feature_importance.values())
        avg_importance = total_importance / len(feature_importance)
        
        noisy_features = []
        for feature, importance in feature_importance.items():
            if importance < avg_importance * 0.01:  # Less than 1% of average
                noisy_features.append(feature)
        
        if noisy_features:
            insights.append({
                "type": "noisy_features",
                "features": noisy_features,
                "avg_importance": avg_importance,
                "insight": f"Found {len(noisy_features)} features with very low importance",
                "recommendation": f"Consider removing noisy features: {', '.join(noisy_features[:5])}",
                "action": "remove_features",
                "priority": "medium"
            })
        
        # Check for balanced importance distribution
        importance_values = list(feature_importance.values())
        importance_std = np.std(importance_values)
        importance_cv = importance_std / np.mean(importance_values) if np.mean(importance_values) > 0 else 0
        
        if importance_cv > 1.0:  # High coefficient of variation
            insights.append({
                "type": "unbalanced_importance",
                "cv": importance_cv,
                "insight": "Feature importance distribution is highly unbalanced",
                "recommendation": "Consider feature engineering to create more balanced features",
                "action": "feature_engineering",
                "priority": "medium"
            })
        
        # Check for feature groups (similar importance)
        if len(sorted_importance) > 3:
            # Group features with similar importance
            feature_groups = []
            current_group = [sorted_importance[0]]
            
            for i in range(1, len(sorted_importance)):
                current_importance = sorted_importance[i][1]
                last_importance = current_group[-1][1]
                
                if abs(current_importance - last_importance) / last_importance < 0.1:  # Within 10%
                    current_group.append(sorted_importance[i])
                else:
                    if len(current_group) > 2:
                        feature_groups.append(current_group)
                    current_group = [sorted_importance[i]]
            
            if len(current_group) > 2:
                feature_groups.append(current_group)
            
            if feature_groups:
                insights.append({
                    "type": "feature_groups",
                    "groups": [[f[0] for f in group] for group in feature_groups],
                    "insight": f"Found {len(feature_groups)} groups of features with similar importance",
                    "recommendation": "Consider creating composite features from these groups",
                    "action": "feature_engineering",
                    "priority": "low"
                })
        
        return insights
    
    def _analyze_data_quality_issues(self, 
                                  X: pd.DataFrame,
                                  y: pd.Series = None,
                                  feature_importance: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Analyze data quality issues"""
        insights = []
        
        # Missing values analysis
        missing_analysis = self._analyze_missing_values(X, feature_importance)
        if missing_analysis:
            insights.extend(missing_analysis)
        
        # Outlier analysis
        outlier_analysis = self._analyze_outliers(X, feature_importance)
        if outlier_analysis:
            insights.extend(outlier_analysis)
        
        # Correlation analysis
        correlation_analysis = self._analyze_correlations(X, feature_importance)
        if correlation_analysis:
            insights.extend(correlation_analysis)
        
        # Target analysis
        if y is not None:
            target_analysis = self._analyze_target_distribution(y)
            if target_analysis:
                insights.extend(target_analysis)
        
        return insights
    
    def _analyze_missing_values(self, 
                              X: pd.DataFrame,
                              feature_importance: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Analyze missing values patterns"""
        insights = []
        
        missing_counts = X.isnull().sum()
        missing_ratios = missing_counts / len(X)
        
        # Features with high missing values
        high_missing = missing_ratios[missing_ratios > 0.3]
        
        if not high_missing.empty:
            high_missing_features = high_missing.index.tolist()
            
            # Check if any high missing features are important
            important_missing = []
            if feature_importance:
                for feature in high_missing_features:
                    if feature in feature_importance and feature_importance[feature] > 0.05:
                        important_missing.append(feature)
            
            insights.append({
                "type": "high_missing_values",
                "features": high_missing_features,
                "missing_ratios": high_missing.to_dict(),
                "important_missing": important_missing,
                "insight": f"Found {len(high_missing_features)} features with >30% missing values",
                "recommendation": "Implement missing value imputation or remove high-missing features",
                "action": "data_cleaning",
                "priority": "high" if important_missing else "medium"
            })
        
        # Features with moderate missing values
        moderate_missing = missing_ratios[(missing_ratios > 0.05) & (missing_ratios <= 0.3)]
        
        if not moderate_missing.empty:
            insights.append({
                "type": "moderate_missing_values",
                "features": moderate_missing.index.tolist(),
                "missing_ratios": moderate_missing.to_dict(),
                "insight": f"Found {len(moderate_missing)} features with 5-30% missing values",
                "recommendation": "Consider imputation strategies for these features",
                "action": "data_cleaning",
                "priority": "medium"
            })
        
        return insights
    
    def _analyze_outliers(self, 
                        X: pd.DataFrame,
                        feature_importance: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Analyze outliers in important features"""
        insights = []
        
        if feature_importance is None:
            return insights
        
        # Focus on top 10 most important features
        top_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        numeric_features = []
        for feature, _ in top_features:
            if feature in X.columns and X[feature].dtype in ['int64', 'float64']:
                numeric_features.append(feature)
        
        outlier_features = []
        for feature in numeric_features:
            if feature in X.columns:
                Q1 = X[feature].quantile(0.25)
                Q3 = X[feature].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = X[(X[feature] < lower_bound) | (X[feature] > upper_bound)]
                outlier_ratio = len(outliers) / len(X)
                
                if outlier_ratio > 0.05:  # More than 5% outliers
                    outlier_features.append({
                        "feature": feature,
                        "outlier_ratio": outlier_ratio,
                        "importance": feature_importance.get(feature, 0),
                        "bounds": {"lower": lower_bound, "upper": upper_bound}
                    })
        
        if outlier_features:
            insights.append({
                "type": "outliers",
                "features": outlier_features,
                "insight": f"Found {len(outlier_features)} important features with significant outliers",
                "recommendation": "Consider outlier treatment (capping, transformation, or removal)",
                "action": "data_cleaning",
                "priority": "medium"
            })
        
        return insights
    
    def _analyze_correlations(self, 
                           X: pd.DataFrame,
                           feature_importance: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Analyze correlations between features"""
        insights = []
        
        # Calculate correlation matrix for numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return insights
        
        correlation_matrix = X[numeric_cols].corr().abs()
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if corr_value > 0.8:  # High correlation threshold
                    feature1 = correlation_matrix.columns[i]
                    feature2 = correlation_matrix.columns[j]
                    
                    # Check importance if available
                    importance1 = feature_importance.get(feature1, 0) if feature_importance else 0
                    importance2 = feature_importance.get(feature2, 0) if feature_importance else 0
                    
                    high_corr_pairs.append({
                        "feature1": feature1,
                        "feature2": feature2,
                        "correlation": corr_value,
                        "importance1": importance1,
                        "importance2": importance2
                    })
        
        if high_corr_pairs:
            insights.append({
                "type": "high_correlations",
                "pairs": high_corr_pairs,
                "insight": f"Found {len(high_corr_pairs)} highly correlated feature pairs",
                "recommendation": "Consider removing redundant features or creating composite features",
                "action": "feature_engineering",
                "priority": "medium"
            })
        
        return insights
    
    def _analyze_target_distribution(self, y: pd.Series) -> List[Dict[str, Any]]:
        """Analyze target variable distribution"""
        insights = []
        
        # Check for class imbalance (classification)
        if y.dtype in ['int64', 'object'] or len(y.unique()) <= 20:
            class_counts = y.value_counts()
            total_samples = len(y)
            
            # Calculate imbalance ratios
            majority_class = class_counts.index[0]
            majority_count = class_counts.iloc[0]
            minority_class = class_counts.index[-1]
            minority_count = class_counts.iloc[-1]
            
            imbalance_ratio = majority_count / minority_count
            
            if imbalance_ratio > 3:  # 3:1 imbalance
                insights.append({
                    "type": "class_imbalance",
                    "majority_class": majority_class,
                    "minority_class": minority_class,
                    "imbalance_ratio": imbalance_ratio,
                    "majority_percentage": (majority_count / total_samples) * 100,
                    "minority_percentage": (minority_count / total_samples) * 100,
                    "insight": f"Significant class imbalance: {imbalance_ratio:.1f}:1 ratio",
                    "recommendation": "Consider class balancing techniques (SMOTE, class weights, or stratified sampling)",
                    "action": "data_preprocessing",
                    "priority": "high"
                })
        
        # Check for target outliers (regression)
        elif y.dtype in ['int64', 'float64']:
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = y[(y < lower_bound) | (y > upper_bound)]
            outlier_ratio = len(outliers) / len(y)
            
            if outlier_ratio > 0.05:  # More than 5% outliers
                insights.append({
                    "type": "target_outliers",
                    "outlier_ratio": outlier_ratio,
                    "bounds": {"lower": lower_bound, "upper": upper_bound},
                    "insight": f"Target variable has {outlier_ratio*100:.1f}% outliers",
                    "recommendation": "Consider outlier treatment or robust modeling approaches",
                    "action": "data_preprocessing",
                    "priority": "medium"
                })
        
        return insights
    
    def _analyze_model_risks(self, 
                           feature_importance: Dict[str, float],
                           X: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze potential model risks"""
        insights = []
        
        # Risk 1: Overfitting to dominant features
        if len(feature_importance) > 1:
            sorted_importance = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            top_feature, top_importance = sorted_importance[0]
            total_importance = sum(feature_importance.values())
            top_feature_ratio = top_importance / total_importance
            
            if top_feature_ratio > 0.5:  # Single feature > 50% of importance
                insights.append({
                    "type": "overfitting_risk",
                    "feature": top_feature,
                    "importance_ratio": top_feature_ratio,
                    "insight": f"Model relies heavily on single feature: {top_feature} ({top_feature_ratio*100:.1f}% of importance)",
                    "recommendation": "Risk of overfitting - consider feature engineering or regularization",
                    "action": "model_improvement",
                    "priority": "high"
                })
        
        # Risk 2: Low feature diversity
        if len(feature_importance) < 5:
            insights.append({
                "type": "low_feature_diversity",
                "feature_count": len(feature_importance),
                "insight": f"Model uses only {len(feature_importance)} features - may lack robustness",
                "recommendation": "Consider adding more diverse features or feature engineering",
                "action": "feature_engineering",
                "priority": "medium"
            })
        
        # Risk 3: High dimensional with sparse importance
        if len(feature_importance) > 50:
            # Check if importance is concentrated in few features
            sorted_importance = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            top_5_importance = sum([imp for _, imp in sorted_importance[:5]])
            total_importance = sum(feature_importance.values())
            concentration_ratio = top_5_importance / total_importance
            
            if concentration_ratio > 0.8:  # Top 5 features > 80% of importance
                insights.append({
                    "type": "sparse_importance",
                    "feature_count": len(feature_importance),
                    "concentration_ratio": concentration_ratio,
                    "insight": f"In high-dimensional space, importance concentrated in few features ({concentration_ratio*100:.1f}% in top 5)",
                    "recommendation": "Risk of poor generalization - consider dimensionality reduction or feature selection",
                    "action": "feature_engineering",
                    "priority": "high"
                })
        
        return insights
    
    def _generate_optimization_suggestions(self, 
                                       feature_importance: Dict[str, float],
                                       X: pd.DataFrame,
                                       explanations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization suggestions"""
        insights = []
        
        # Suggestion 1: Feature engineering opportunities
        if len(feature_importance) > 0:
            # Look for low-importance features that could be combined
            sorted_importance = sorted(
                feature_importance.items(), 
                key=lambda x: x[1]
            )
            
            low_importance_features = [f for f, imp in sorted_importance[:10] if imp < 0.05]
            
            if len(low_importance_features) > 2:
                insights.append({
                    "type": "feature_engineering_opportunity",
                    "features": low_importance_features,
                    "insight": f"Found {len(low_importance_features)} low-importance features that could be combined",
                    "recommendation": "Create interaction features or polynomial features from low-importance variables",
                    "action": "feature_engineering",
                    "priority": "medium"
                })
        
        # Suggestion 2: Feature selection opportunities
        total_features = len(feature_importance)
        if total_features > 20:
            # Calculate cumulative importance
            sorted_importance = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            cumulative_importance = 0
            features_to_keep = []
            
            for feature, importance in sorted_importance:
                cumulative_importance += importance
                features_to_keep.append(feature)
                
                if cumulative_importance >= 0.95:  # 95% of total importance
                    break
            
            if len(features_to_keep) < total_features:
                insights.append({
                    "type": "feature_selection_opportunity",
                    "features_to_keep": features_to_keep,
                    "features_to_remove": total_features - len(features_to_keep),
                    "cumulative_importance": cumulative_importance,
                    "insight": f"Can remove {total_features - len(features_to_keep)} features while keeping 95% importance",
                    "recommendation": f"Keep top {len(features_to_keep)} features for more efficient model",
                    "action": "feature_selection",
                    "priority": "low"
                })
        
        # Suggestion 3: Model type optimization
        shap_explanations = explanations.get("shap_explanations", {})
        if shap_explanations:
            # Check for non-linear patterns
            insights.append({
                "type": "model_optimization",
                "insight": "SHAP analysis available for model optimization",
                "recommendation": "Use SHAP dependence plots to identify non-linear relationships and feature interactions",
                "action": "model_analysis",
                "priority": "medium"
            })
        
        return insights
    
    def _generate_business_insights(self, 
                                 feature_importance: Dict[str, float],
                                 X: pd.DataFrame,
                                 y: pd.Series = None) -> List[Dict[str, Any]]:
        """Generate business-focused insights"""
        insights = []
        
        # Get top features
        sorted_importance = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Business insight 1: Key drivers
        top_5_features = sorted_importance[:5]
        
        insights.append({
            "type": "key_drivers",
            "features": top_5_features,
            "insight": "Top 5 most important features identified",
            "recommendation": "Focus business efforts and data collection on these key drivers",
            "action": "business_strategy",
            "priority": "high"
        })
        
        # Business insight 2: Data collection opportunities
        if len(X.columns) > 10:
            # Look for data types that might need more features
            numeric_features = X.select_dtypes(include=[np.number]).columns
            categorical_features = X.select_dtypes(include=['object']).columns
            
            if len(categorical_features) == 0 and len(numeric_features) > 5:
                insights.append({
                    "type": "data_collection_opportunity",
                    "insight": "Model uses only numeric features - categorical features might add value",
                    "recommendation": "Consider collecting categorical data for additional business insights",
                    "action": "data_strategy",
                    "priority": "medium"
                })
        
        # Business insight 3: Cost-benefit analysis
        if len(sorted_importance) > 0:
            # Estimate feature collection/processing costs vs importance
            high_importance = sum([imp for _, imp in sorted_importance[:3] if isinstance(imp, (int, float))])
            total_importance = sum([imp for imp in feature_importance.values() if isinstance(imp, (int, float))])
            
            if total_importance > 0 and high_importance / total_importance > 0.7:  # Top 3 > 70%
                insights.append({
                    "type": "cost_benefit_insight",
                    "concentration_ratio": high_importance / total_importance,
                    "insight": "Model performance driven by few key features",
                    "recommendation": "Ensure data quality and collection processes for these critical features",
                    "action": "data_quality",
                    "priority": "high"
                })
        
        return insights
    
    def _calculate_action_priorities(self, actionable_insights: Dict[str, Any]) -> Dict[str, int]:
        """Calculate priority levels for different types of actions"""
        priorities = {
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        # Count priorities from all insight types
        all_insights = []
        for insight_type in actionable_insights.values():
            if isinstance(insight_type, list):
                all_insights.extend(insight_type)
        
        for insight in all_insights:
            priority = insight.get("priority", "medium")
            priorities[priority] += 1
        
        return priorities
    
    def get_actionable_summary(self, actionable_insights: Dict[str, Any]) -> str:
        """Generate human-readable summary of actionable insights"""
        try:
            summary_lines = []
            
            summary_lines.append("🎯 ACTIONABLE EXPLAINABILITY SUMMARY")
            summary_lines.append("=" * 50)
            
            # Priority counts
            priorities = actionable_insights.get("action_priority", {})
            summary_lines.append(f"\n🚨 Priority Breakdown:")
            summary_lines.append(f"   High Priority: {priorities.get('high', 0)} actions")
            summary_lines.append(f"   Medium Priority: {priorities.get('medium', 0)} actions")
            summary_lines.append(f"   Low Priority: {priorities.get('low', 0)} actions")
            
            # Feature recommendations
            feature_recs = actionable_insights.get("feature_recommendations", [])
            if feature_recs:
                summary_lines.append(f"\n🔧 Feature Recommendations:")
                for rec in feature_recs:
                    priority_emoji = "🚨" if rec["priority"] == "high" else "⚠️" if rec["priority"] == "medium" else "💡"
                    summary_lines.append(f"   {priority_emoji} {rec['insight']}")
                    summary_lines.append(f"      💬 {rec['recommendation']}")
            
            # Data quality issues
            quality_issues = actionable_insights.get("data_quality_issues", [])
            if quality_issues:
                summary_lines.append(f"\n📊 Data Quality Issues:")
                for issue in quality_issues:
                    priority_emoji = "🚨" if issue["priority"] == "high" else "⚠️" if issue["priority"] == "medium" else "💡"
                    summary_lines.append(f"   {priority_emoji} {issue['insight']}")
                    summary_lines.append(f"      💬 {issue['recommendation']}")
            
            # Model risks
            model_risks = actionable_insights.get("model_risks", [])
            if model_risks:
                summary_lines.append(f"\n⚠️ Model Risks:")
                for risk in model_risks:
                    priority_emoji = "🚨" if risk["priority"] == "high" else "⚠️" if risk["priority"] == "medium" else "💡"
                    summary_lines.append(f"   {priority_emoji} {risk['insight']}")
                    summary_lines.append(f"      💬 {risk['recommendation']}")
            
            # Business insights
            business_insights = actionable_insights.get("business_insights", [])
            if business_insights:
                summary_lines.append(f"\n💼 Business Insights:")
                for insight in business_insights:
                    priority_emoji = "🚨" if insight["priority"] == "high" else "⚠️" if insight["priority"] == "medium" else "💡"
                    summary_lines.append(f"   {priority_emoji} {insight['insight']}")
                    summary_lines.append(f"      💬 {insight['recommendation']}")
            
            # Optimization suggestions
            opt_suggestions = actionable_insights.get("optimization_suggestions", [])
            if opt_suggestions:
                summary_lines.append(f"\n🚀 Optimization Suggestions:")
                for suggestion in opt_suggestions:
                    priority_emoji = "🚨" if suggestion["priority"] == "high" else "⚠️" if suggestion["priority"] == "medium" else "💡"
                    summary_lines.append(f"   {priority_emoji} {suggestion['insight']}")
                    summary_lines.append(f"      💬 {suggestion['recommendation']}")
            
            summary_lines.append(f"\n📋 Next Steps:")
            high_priority_count = priorities.get('high', 0)
            if high_priority_count > 0:
                summary_lines.append(f"   1. Address {high_priority_count} high-priority issues immediately")
            summary_lines.append("   2. Review medium-priority recommendations for quick wins")
            summary_lines.append("   3. Consider low-priority optimizations for long-term improvement")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Failed to generate actionable summary: {e}")
            return f"Error generating summary: {e}"


# Convenience functions
def generate_actionable_insights(explanations: Dict[str, Any], 
                               X: pd.DataFrame,
                               y: pd.Series = None) -> Dict[str, Any]:
    """Convenience function for generating actionable insights"""
    actionable = ActionableExplainability()
    return actionable.generate_actionable_insights(explanations, X, y)


def get_actionable_summary(actionable_insights: Dict[str, Any]) -> str:
    """Convenience function for getting actionable summary"""
    actionable = ActionableExplainability()
    return actionable.get_actionable_summary(actionable_insights)
