"""
Model Explainability Layer
Feature importance, SHAP values, and interpretability analysis
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

logger = logging.getLogger(__name__)


class ModelExplainability:
    """
    Comprehensive model explainability and interpretability analysis
    """
    
    def __init__(self,
                 use_shap: bool = True,
                 shap_sample_size: int = None,
                 feature_names: Optional[List[str]] = None,
                 random_state: int = None):
        """
        Initialize model explainability
        
        Args:
            use_shap: Whether to use SHAP for explanations
            shap_sample_size: Number of samples for SHAP analysis
            feature_names: Names of features
            random_state: Random seed
        """
        # Use configurable defaults
        try:
            from config.settings import get_config_value
            shap_sample_size = shap_sample_size or get_config_value('explainability', 'shap_sample_size', 100)
            random_state = random_state if random_state is not None else get_config_value('explainability', 'random_state', 42)
        except ImportError:
            # Fallback to hardcoded values
            shap_sample_size = shap_sample_size or 100
            random_state = random_state if random_state is not None else 42
        
        self.use_shap = use_shap and SHAP_AVAILABLE
        self.shap_sample_size = shap_sample_size
        self.feature_names = feature_names
        self.random_state = random_state
        
        # Explainability results
        self.feature_importance = {}
        self.shap_values = None
        self.shap_explainer = None
        self.explanation_metadata = {}
        
        if self.use_shap:
            logger.info("SHAP explainability enabled")
        else:
            logger.info("Using basic feature importance only (SHAP not available)")
    
    def explain_model(self, 
                     model: BaseEstimator,
                     X: Union[np.ndarray, pd.DataFrame],
                     y: Optional[Union[np.ndarray, pd.Series]] = None,
                     task_type: str = "classification") -> Dict[str, Any]:
        """
        Generate comprehensive model explanations
        
        Args:
            model: Trained model
            X: Feature data
            y: Target data (optional)
            task_type: "classification" or "regression"
            
        Returns:
            Explanation results
        """
        logger.info("Generating model explanations...")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            if self.feature_names:
                df = pd.DataFrame(X, columns=self.feature_names)
            else:
                df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            df = X.copy()
        
        # Generate feature importance
        feature_importance = self._get_feature_importance(model, df, task_type)
        
        # Generate SHAP explanations if available
        shap_explanations = {}
        if self.use_shap:
            try:
                shap_explanations = self._generate_shap_explanations(model, df, task_type)
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")
                shap_explanations = {"error": str(e)}
        
        # Generate interpretability report
        interpretability_report = self._generate_interpretability_report(
            feature_importance, shap_explanations, task_type
        )
        
        # Compile results
        explanations = {
            "feature_importance": feature_importance,
            "shap_explanations": shap_explanations,
            "interpretability_report": interpretability_report,
            "task_type": task_type,
            "n_features": len(df.columns),
            "feature_names": list(df.columns),
            "explanation_metadata": {
                "shap_used": self.use_shap,
                "shap_available": SHAP_AVAILABLE,
                "sample_size": self.shap_sample_size if self.use_shap else None
            }
        }
        
        self.explanation_metadata = explanations
        
        logger.info(f"Model explanations generated: {len(feature_importance)} features analyzed")
        
        return explanations
    
    def _get_feature_importance(self, 
                              model: BaseEstimator,
                              df: pd.DataFrame,
                              task_type: str) -> Dict[str, Any]:
        """
        Get feature importance from model
        
        Args:
            model: Trained model
            df: Feature DataFrame
            task_type: Type of task
            
        Returns:
            Feature importance dictionary
        """
        importance_results = {}
        
        try:
            # Method 1: Direct feature_importances_
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_results["direct"] = dict(zip(df.columns, importances))
            
            # Method 2: Coefficients (for linear models)
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if coef.ndim > 1:
                    # Multi-class case - use absolute values
                    coef = np.abs(coef).mean(axis=0)
                importance_results["coefficients"] = dict(zip(df.columns, np.abs(coef)))
            
            # Method 3: Permutation importance
            try:
                from sklearn.inspection import permutation_importance
                
                # Create a simple scoring function
                if task_type == "classification":
                    from sklearn.metrics import accuracy_score
                    scoring_func = accuracy_score
                else:
                    from sklearn.metrics import r2_score
                    scoring_func = r2_score
                
                # Calculate permutation importance
                result = permutation_importance(
                    model, df, y if 'y' in locals() else None,
                    scoring='accuracy' if task_type == "classification" else 'r2',
                    n_repeats=5,
                    random_state=self.random_state
                )
                
                importance_results["permutation"] = dict(zip(df.columns, result.importances_mean))
                
            except Exception as e:
                logger.debug(f"Permutation importance failed: {e}")
            
            # Aggregate importance (average across methods)
            if len(importance_results) > 1:
                # Normalize each method
                normalized_importances = {}
                for method, importance_dict in importance_results.items():
                    values = np.array(list(importance_dict.values()))
                    if values.std() > 0:
                        normalized = (values - values.min()) / (values.max() - values.min())
                    else:
                        normalized = np.ones_like(values)
                    normalized_importances[method] = dict(zip(importance_dict.keys(), normalized))
                
                # Average across methods
                avg_importance = defaultdict(float)
                count = defaultdict(int)
                
                for method, importance_dict in normalized_importances.items():
                    for feature, value in importance_dict.items():
                        avg_importance[feature] += value
                        count[feature] += 1
                
                # Calculate average
                final_importance = {
                    feature: avg_importance[feature] / count[feature]
                    for feature in avg_importance
                }
                
                importance_results["aggregated"] = final_importance
            
            # Sort by importance
            for method in importance_results:
                if isinstance(importance_results[method], dict):
                    importance_results[method] = dict(
                        sorted(importance_results[method].items(), 
                              key=lambda x: x[1], reverse=True)
                    )
            
        except Exception as e:
            logger.error(f"Feature importance extraction failed: {e}")
            importance_results = {"error": str(e)}
        
        return importance_results
    
    def _generate_shap_explanations(self, 
                                   model: BaseEstimator,
                                   df: pd.DataFrame,
                                   task_type: str) -> Dict[str, Any]:
        """
        Generate SHAP explanations
        
        Args:
            model: Trained model
            df: Feature DataFrame
            task_type: Type of task
            
        Returns:
            SHAP explanations
        """
        if not SHAP_AVAILABLE:
            return {"error": "SHAP not available"}
        
        try:
            # Sample data for SHAP analysis
            if len(df) > self.shap_sample_size:
                df_sample = df.sample(n=self.shap_sample_size, random_state=self.random_state)
            else:
                df_sample = df
            
            X_sample = df_sample.values
            
            # Choose appropriate explainer
            explainer_type = self._choose_shap_explainer(model, df_sample)
            
            if explainer_type == "tree":
                try:
                    self.shap_explainer = shap.TreeExplainer(model, check_additivity=False)
                except Exception as e:
                    logger.warning(f"TreeExplainer failed, trying with intervention: {e}")
                    try:
                        self.shap_explainer = shap.TreeExplainer(model, feature_perturbation='interventional', check_additivity=False)
                    except Exception as e2:
                        logger.error(f"TreeExplainer with intervention failed: {e2}")
                        raise e2
            elif explainer_type == "linear":
                self.shap_explainer = shap.LinearExplainer(model, df_sample)
            elif explainer_type == "kernel":
                self.shap_explainer = shap.KernelExplainer(
                    model.predict, shap.sample(df_sample, 100), random_state=self.random_state
                )
            else:
                # Default to Permutation explainer
                self.shap_explainer = shap.PermutationExplainer(
                    model.predict, df_sample, random_state=self.random_state
                )
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Handle multi-output cases
            if isinstance(shap_values, list):
                # Multi-class classification
                shap_values_mean = np.abs([np.mean(sv, axis=0) for sv in shap_values]).mean(axis=0)
            else:
                shap_values_mean = np.abs(shap_values).mean(axis=0)
            
            # Create feature importance from SHAP
            shap_importance = dict(zip(df_sample.columns, shap_values_mean))
            shap_importance = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))
            
            # Generate summary statistics
            shap_summary = {
                "mean_abs_shap": shap_values_mean,
                "feature_importance": shap_importance,
                "explainer_type": explainer_type,
                "sample_size": len(df_sample)
            }
            
            # Generate feature interactions (if possible)
            interactions = self._analyze_shap_interactions(shap_values, df_sample)
            
            return {
                "summary": shap_summary,
                "interactions": interactions,
                "values": shap_values,
                "explainer": str(type(self.shap_explainer).__name__)
            }
            
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            return {"error": str(e)}
    
    def _choose_shap_explainer(self, model: BaseEstimator, df: pd.DataFrame) -> str:
        """
        Choose appropriate SHAP explainer
        
        Args:
            model: Trained model
            df: Feature DataFrame
            
        Returns:
            Explainer type
        """
        model_type = type(model).__name__.lower()
        
        # Tree-based models
        if any(tree_type in model_type for tree_type in ['randomforest', 'xgboost', 'lightgbm', 'gradientboosting', 'decisiontree']):
            return "tree"
        
        # Linear models
        elif any(linear_type in model_type for linear_type in ['linear', 'logistic', 'ridge', 'lasso', 'svm']):
            return "linear"
        
        # Neural networks
        elif 'mlp' in model_type or 'neural' in model_type:
            return "kernel"
        
        # Default
        else:
            return "permutation"
    
    def _analyze_shap_interactions(self, 
                                  shap_values: np.ndarray,
                                  df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze SHAP feature interactions
        
        Args:
            shap_values: SHAP values
            df: Feature DataFrame
            
        Returns:
            Interaction analysis
        """
        try:
            if len(df.columns) > 10:  # Skip for high-dimensional data
                return {"skipped": "Too many features for interaction analysis"}
            
            # Calculate interaction effects (simplified)
            interactions = {}
            
            # Get top features
            if isinstance(shap_values, list):
                # Multi-class case
                shap_mean = np.abs([np.mean(sv, axis=0) for sv in shap_values]).mean(axis=0)
            else:
                shap_mean = np.abs(shap_values).mean(axis=0)
            
            top_features = np.argsort(shap_mean)[-5:]  # Top 5 features
            
            # Calculate pairwise interactions
            for i, feat1 in enumerate(top_features):
                for feat2 in top_features[i+1:]:
                    feat1_name = df.columns[feat1]
                    feat2_name = df.columns[feat2]
                    
                    # Simple interaction: correlation of SHAP values
                    if isinstance(shap_values, list):
                        # Multi-class case
                        shap_feat1 = np.mean([sv[:, feat1] for sv in shap_values], axis=0)
                        shap_feat2 = np.mean([sv[:, feat2] for sv in shap_values], axis=0)
                    else:
                        shap_feat1 = shap_values[:, feat1]
                        shap_feat2 = shap_values[:, feat2]
                    
                    correlation = np.corrcoef(shap_feat1, shap_feat2)[0, 1]
                    
                    interactions[f"{feat1_name}_x_{feat2_name}"] = {
                        "correlation": correlation,
                        "strength": abs(correlation)
                    }
            
            # Sort by interaction strength
            interactions = dict(
                sorted(interactions.items(), 
                      key=lambda x: x[1]["strength"], reverse=True)
            )
            
            return interactions
            
        except Exception as e:
            logger.debug(f"SHAP interaction analysis failed: {e}")
            return {"error": str(e)}
    
    def _generate_interpretability_report(self, 
                                         feature_importance: Dict[str, Any],
                                         shap_explanations: Dict[str, Any],
                                         task_type: str) -> Dict[str, Any]:
        """
        Generate comprehensive interpretability report
        
        Args:
            feature_importance: Feature importance results
            shap_explanations: SHAP explanations
            task_type: Type of task
            
        Returns:
            Interpretability report
        """
        report = {
            "summary": {},
            "top_features": {},
            "insights": [],
            "recommendations": []
        }
        
        try:
            # Get primary importance method
            if "aggregated" in feature_importance:
                primary_importance = feature_importance["aggregated"]
            elif "direct" in feature_importance:
                primary_importance = feature_importance["direct"]
            elif "permutation" in feature_importance:
                primary_importance = feature_importance["permutation"]
            else:
                primary_importance = {}
            
            # Top features
            if primary_importance:
                top_features = list(primary_importance.items())[:10]
                report["top_features"] = {
                    "top_10": top_features,
                    "top_5": top_features[:5],
                    "most_important": top_features[0] if top_features else None
                }
            
            # Summary statistics
            if primary_importance:
                values = list(primary_importance.values())
                report["summary"] = {
                    "total_features": len(primary_importance),
                    "mean_importance": np.mean(values),
                    "std_importance": np.std(values),
                    "max_importance": max(values),
                    "min_importance": min(values),
                    "importance_distribution": {
                        "high": len([v for v in values if v > 0.7]),
                        "medium": len([v for v in values if 0.3 <= v <= 0.7]),
                        "low": len([v for v in values if v < 0.3])
                    }
                }
            
            # Generate insights
            insights = []
            
            if primary_importance:
                # Feature dominance
                if len(primary_importance) > 1:
                    top_importance = list(primary_importance.values())[0]
                    second_importance = list(primary_importance.values())[1]
                    if top_importance > second_importance * 2:
                        insights.append("One feature dominates the model - consider feature engineering")
                
                # Feature distribution
                high_importance = len([v for v in primary_importance.values() if v > 0.5])
                if high_importance > 5:
                    insights.append("Many features have high importance - model might be complex")
                elif high_importance < 2:
                    insights.append("Few features drive predictions - model is interpretable")
            
            # SHAP-specific insights
            if shap_explanations and "summary" in shap_explanations:
                shap_summary = shap_explanations["summary"]
                
                if "explainer_type" in shap_summary:
                    explainer_type = shap_summary["explainer_type"]
                    insights.append(f"SHAP analysis using {explainer_type} explainer")
                
                if "interactions" in shap_explanations and shap_explanations["interactions"]:
                    interactions = shap_explanations["interactions"]
                    if isinstance(interactions, dict) and len(interactions) > 0:
                        top_interaction = list(interactions.items())[0]
                        insights.append(f"Strongest interaction: {top_interaction[0]} (strength: {top_interaction[1]['strength']:.3f})")
            
            report["insights"] = insights
            
            # Generate recommendations
            recommendations = []
            
            if primary_importance:
                # Feature selection recommendations
                low_importance = len([v for v in primary_importance.values() if v < 0.1])
                if low_importance > len(primary_importance) * 0.5:
                    recommendations.append("Consider removing low-importance features to simplify model")
                
                # Feature engineering recommendations
                if len(primary_importance) < 10:
                    recommendations.append("Consider creating interaction features to improve model performance")
            
            # SHAP recommendations
            if shap_explanations and "summary" in shap_explanations:
                recommendations.append("Use SHAP values for detailed feature-level explanations")
                recommendations.append("Consider SHAP dependence plots for non-linear relationships")
            
            if task_type == "classification":
                recommendations.append("Monitor feature importance for model fairness and bias")
            else:
                recommendations.append("Use feature importance to understand key drivers of predictions")
            
            report["recommendations"] = recommendations
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            report["error"] = str(e)
        
        return report
    
    def get_explanation_summary(self) -> str:
        """
        Get human-readable explanation summary
        
        Returns:
            Formatted explanation summary
        """
        if not self.explanation_metadata:
            return "No explanations available. Run explain_model() first."
        
        summary = []
        summary.append("🔍 MODEL EXPLAINABILITY REPORT")
        summary.append("=" * 50)
        
        # Basic info
        metadata = self.explanation_metadata.get("explanation_metadata", {})
        summary.append(f"Task Type: {self.explanation_metadata.get('task_type', 'Unknown')}")
        summary.append(f"Features: {self.explanation_metadata.get('n_features', 0)}")
        summary.append(f"SHAP Used: {metadata.get('shap_used', False)}")
        summary.append("")
        
        # Top features
        top_features = self.explanation_metadata.get("interpretability_report", {}).get("top_features", {})
        if "top_5" in top_features:
            summary.append("🏆 TOP 5 MOST IMPORTANT FEATURES:")
            for i, (feature, importance) in enumerate(top_features["top_5"], 1):
                summary.append(f"{i}. {feature} → {importance:.4f}")
            summary.append("")
        
        # Insights
        insights = self.explanation_metadata.get("interpretability_report", {}).get("insights", [])
        if insights:
            summary.append("💡 KEY INSIGHTS:")
            for insight in insights:
                summary.append(f"• {insight}")
            summary.append("")
        
        # Recommendations
        recommendations = self.explanation_metadata.get("interpretability_report", {}).get("recommendations", [])
        if recommendations:
            summary.append("📋 RECOMMENDATIONS:")
            for rec in recommendations:
                summary.append(f"• {rec}")
        
        return "\n".join(summary)
    
    def plot_feature_importance(self, 
                              top_n: int = 10,
                              save_path: Optional[str] = None,
                              method: str = "aggregated") -> None:
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to show
            save_path: Path to save plot
            method: Importance method to use
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            feature_importance = self.explanation_metadata.get("feature_importance", {})
            
            if method not in feature_importance:
                # Try alternative methods
                for alt_method in ["aggregated", "direct", "permutation", "coefficients"]:
                    if alt_method in feature_importance:
                        method = alt_method
                        break
            
            if method not in feature_importance:
                logger.error("No feature importance data available")
                return
            
            importance_dict = feature_importance[method]
            
            # Get top features
            top_features = list(importance_dict.items())[:top_n]
            features, importances = zip(*top_features)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(importances), y=list(features))
            plt.title(f'Top {top_n} Feature Importance ({method})')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to plot feature importance: {e}")
    
    def plot_shap_summary(self, 
                         save_path: Optional[str] = None,
                         max_display: int = 20) -> None:
        """
        Plot SHAP summary plot
        
        Args:
            save_path: Path to save plot
            max_display: Maximum number of features to display
        """
        try:
            if not SHAP_AVAILABLE or not self.shap_explainer:
                logger.error("SHAP not available or not calculated")
                return
            
            shap_explanations = self.explanation_metadata.get("shap_explanations", {})
            if "values" not in shap_explanations:
                logger.error("SHAP values not available")
                return
            
            # Create summary plot
            shap.summary_plot(
                shap_explanations["values"],
                feature_names=self.explanation_metadata.get("feature_names", []),
                max_display=max_display,
                show=True
            )
            
            logger.info("SHAP summary plot displayed")
            
        except Exception as e:
            logger.error(f"Failed to plot SHAP summary: {e}")


def explain_model(model: BaseEstimator,
                X: Union[np.ndarray, pd.DataFrame],
                y: Optional[Union[np.ndarray, pd.Series]] = None,
                task_type: str = "classification",
                **kwargs) -> Dict[str, Any]:
    """
    Convenience function for model explainability
    
    Args:
        model: Trained model
        X: Feature data
        y: Target data
        task_type: Type of task
        **kwargs: Additional arguments for ModelExplainability
        
    Returns:
        Explanation results
    """
    explainer = ModelExplainability(**kwargs)
    return explainer.explain_model(model, X, y, task_type)
