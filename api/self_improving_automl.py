"""
Self-Improving AutoML System
Enhanced AutoML with pattern learning and actionable insights
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from .automl import AutoML

# Import meta-learning components with fallback handling
try:
    from ..features.meta_learning.pattern_learner import PatternLearner
except ImportError:
    try:
        from features.meta_learning.pattern_learner import PatternLearner
    except ImportError:
        PatternLearner = None
    logging.warning("PatternLearner not available, self-improving features limited")

try:
    from ..intelligence.meta_optimizer import MetaOptimizer
except ImportError:
    MetaOptimizer = None
    logging.warning("MetaOptimizer not available, self-improving features limited")

try:
    from ..features.explainability.actionable_explainability import ActionableExplainability
except ImportError:
    ActionableExplainability = None
    logging.warning("ActionableExplainability not available, insights limited")

try:
    from ..features.explainability.decision_explainer import DecisionExplainer
except ImportError:
    DecisionExplainer = None
    logging.warning("DecisionExplainer not available, explanations limited")

logger = logging.getLogger(__name__)


class SelfImprovingAutoML(AutoML):
    """
    Enhanced AutoML system that learns from past experiments and provides actionable insights
    """
    
    def __init__(self, 
                 n_trials=50, 
                 timeout=None, 
                 cv=3,
                 use_adaptive_optimization=True,
                 use_dataset_optimization=True,
                 use_caching=True,
                 show_progress=True,
                 use_explainability=True,
                 enable_pattern_learning=True,
                 enable_actionable_insights=True):
        """
        Initialize self-improving AutoML
        
        Args:
            enable_pattern_learning: Enable pattern learning from experiments
            enable_actionable_insights: Enable actionable explainability insights
        """
        # Initialize parent AutoML
        super().__init__(
            n_trials=n_trials,
            timeout=timeout,
            cv=cv,
            use_adaptive_optimization=use_adaptive_optimization,
            use_dataset_optimization=use_dataset_optimization,
            use_caching=use_caching,
            show_progress=show_progress,
            use_explainability=use_explainability
        )
        
        # Self-improving features
        self.enable_pattern_learning = enable_pattern_learning
        self.enable_actionable_insights = enable_actionable_insights
        
        # Initialize self-improving components
        self.pattern_learner = PatternLearner() if (enable_pattern_learning and PatternLearner is not None) else None
        self.meta_optimizer = MetaOptimizer() if (MetaOptimizer is not None) else None
        self.actionable_explainability = ActionableExplainability() if (ActionableExplainability is not None) else None
        self.decision_explainer = DecisionExplainer() if (DecisionExplainer is not None) else None
        
        # Learned patterns storage
        self.learned_patterns = {}
        self.intelligent_recommendations = {}
        self._last_actionable_insights = {}
        
        logger.info(f"Self-improving AutoML initialized with pattern learning: {enable_pattern_learning}")
    
    def fit_with_learning(self, X, y, store_experiment: bool = True):
        """
        Enhanced fit with pattern learning and actionable insights
        
        Args:
            X: Feature data
            y: Target data
            store_experiment: Whether to store this experiment for pattern learning
            
        Returns:
            Self with learned patterns and insights
        """
        try:
            # Call parent fit method
            super().fit(X, y)
            
            # Store experiment for pattern learning
            if store_experiment and self.enable_pattern_learning:
                experiment_result = self._create_experiment_result(X, y)
                self.learned_patterns = self.learn_from_experiment(experiment_result)
                logger.info(f"Stored experiment for pattern learning: {len(self.learned_patterns)} insights")
            
            # Generate intelligent recommendations
            if self.enable_pattern_learning:
                dataset_info = {
                    "n_samples": len(X),
                    "n_features": X.shape[1] if hasattr(X, 'shape') else len(X[0]),
                    "data_type": getattr(self, 'primary_data_type', 'unknown')
                }
                self.intelligent_recommendations = self.get_intelligent_recommendations(dataset_info)
                logger.info(f"Generated {len(self.intelligent_recommendations.get('models', []))} intelligent recommendations")
            
            return self
            
        except Exception as e:
            logger.error(f"Self-improving fit failed: {e}")
            raise
    
    def explain_with_actions(self, 
                          X: Union[np.ndarray, pd.DataFrame],
                          y: Optional[Union[np.ndarray, pd.Series]] = None,
                          use_shap: bool = True,
                          top_n: int = 10) -> Dict[str, Any]:
        """
        Generate explanations with actionable insights
        
        Args:
            X: Feature data
            y: Target data (optional)
            use_shap: Whether to use SHAP explanations
            top_n: Number of top features to return
            
        Returns:
            Enhanced explanations with actionable insights
        """
        try:
            # Generate standard explanations
            explanations = super().explain(X, y, use_shap, top_n)
            
            # Add actionable insights if enabled
            if self.enable_actionable_insights:
                actionable_insights = self.get_actionable_explanations(X, y)
                explanations["actionable_insights"] = actionable_insights
                explanations["actionable_summary"] = self.get_actionable_summary()
                
                logger.info(f"Generated {len(actionable_insights.get('feature_recommendations', []))} actionable insights")
            
            # Add intelligent recommendations
            if self.enable_pattern_learning:
                explanations["intelligent_recommendations"] = self.intelligent_recommendations
                
                # Add pattern-based optimization suggestions
                if self.intelligent_recommendations:
                    explanations["pattern_based_optimization"] = self._get_pattern_optimization_suggestions()
            
            return explanations
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced explanations: {e}")
            return {"error": str(e)}
    
    def optimize_with_patterns(self, X, y):
        """
        Optimize search space based on learned patterns
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Optimization results with pattern-based modifications
        """
        try:
            # Get dataset info
            dataset_info = {
                "n_samples": len(X),
                "n_features": X.shape[1] if hasattr(X, 'shape') else len(X[0]),
                "data_type": getattr(self, 'primary_data_type', 'unknown')
            }
            
            # Get intelligent recommendations
            recommendations = self.get_intelligent_recommendations(dataset_info)
            
            # Optimize search space
            optimized_space = self._optimize_search_space_with_patterns(recommendations)
            
            # Apply optimized configuration
            if optimized_space.get("modifications"):
                logger.info(f"Applied {len(optimized_space['modifications'])} pattern-based optimizations")
                
                # Temporarily modify AutoML configuration
                original_n_trials = self.n_trials
                original_adaptive = self.use_adaptive_optimization
                
                # Apply modifications
                if "trial_limit" in optimized_space.get("optimized_space", {}):
                    self.n_trials = optimized_space["optimized_space"]["trial_limit"]
                    logger.info(f"Reduced trials from {original_n_trials} to {self.n_trials} based on patterns")
                
                if "use_adaptive" in optimized_space.get("optimized_space", {}):
                    self.use_adaptive_optimization = optimized_space["optimized_space"]["use_adaptive"]
                    logger.info(f"Set adaptive optimization to {self.use_adaptive_optimization} based on patterns")
            
            # Fit with optimized configuration
            result = self.fit_with_learning(X, y)
            
            # Restore original configuration
            self.n_trials = original_n_trials
            self.use_adaptive_optimization = original_adaptive
            
            result["optimization_applied"] = optimized_space
            result["pattern_recommendations"] = recommendations
            
            return result
            
        except Exception as e:
            logger.error(f"Pattern-based optimization failed: {e}")
            raise
    
    def run_comprehensive_analysis(self, X, y):
        """
        Run comprehensive analysis with benchmarking and insights
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Comprehensive analysis results
        """
        try:
            analysis_results = {
                "model_performance": {},
                "actionable_insights": {},
                "pattern_recommendations": {},
                "benchmarking": {},
                "overall_assessment": {}
            }
            
            # Train model with learning
            self.fit_with_learning(X, y)
            
            # Generate actionable insights
            if self.enable_actionable_insights:
                actionable_insights = self.get_actionable_explanations(X, y)
                self._last_actionable_insights = actionable_insights  # Store for summary
                analysis_results["actionable_insights"] = actionable_insights
                
                # Generate summary
                summary = self.get_actionable_summary()
                analysis_results["actionable_summary"] = summary
            
            # Get pattern recommendations
            if self.enable_pattern_learning:
                dataset_info = {
                    "n_samples": len(X),
                    "n_features": X.shape[1] if hasattr(X, 'shape') else len(X[0]),
                    "data_type": getattr(self, 'primary_data_type', 'unknown')
                }
                recommendations = self.get_intelligent_recommendations(dataset_info)
                analysis_results["pattern_recommendations"] = recommendations
            
            # Run enhanced benchmarking
            try:
                from ..benchmarking.enhanced_benchmarking import EnhancedBenchmarking
                benchmarking = EnhancedBenchmarking()
                benchmark_results = benchmarking.run_comprehensive_benchmark()
                analysis_results["benchmarking"] = benchmark_results
                
                logger.info(f"Completed comprehensive benchmarking: {len(benchmark_results.get('datasets', []))} datasets")
                
            except Exception as e:
                logger.warning(f"Benchmarking failed: {e}")
                analysis_results["benchmarking"] = {"error": str(e)}
            
            # Generate overall assessment
            analysis_results["overall_assessment"] = self._generate_overall_assessment(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            raise
    
    def _create_experiment_result(self, X, y) -> Dict[str, Any]:
        """Create experiment result for pattern learning"""
        try:
            # Get model performance
            if self.best_pipeline is None:
                return {"error": "No trained model available"}
            
            # Make predictions on training data for performance estimation
            y_pred = self.predict(X)
            
            # Calculate performance metrics
            if self.task_type == "classification":
                performance = {
                    "accuracy": accuracy_score(y, y_pred),
                    "f1_score": f1_score(y, y_pred, average='weighted')
                }
            else:
                performance = {
                    "mse": mean_squared_error(y, y_pred),
                    "r2_score": r2_score(y, y_pred)
                }
            
            # Create experiment result
            experiment_result = {
                "dataset_info": {
                    "n_samples": len(X),
                    "n_features": X.shape[1] if hasattr(X, 'shape') else len(X[0]),
                    "data_type": getattr(self, 'primary_data_type', 'unknown')
                },
                "model_info": {
                    "name": self._get_model_name(),
                    "task_type": self.task_type
                },
                "performance_metrics": performance,
                "optimization_info": {
                    "n_trials": getattr(self, 'optimization_metadata', {}).get('n_trials', self.n_trials),
                    "strategy": "adaptive" if self.use_adaptive_optimization else "standard"
                },
                "feature_engineering": {
                    "enabled": True,
                    "types": ["smart", "data_type_aware"]
                },
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            return experiment_result
            
        except Exception as e:
            logger.error(f"Failed to create experiment result: {e}")
            return {"error": str(e)}
    
    def _get_model_name(self) -> str:
        """Get the name of the best model"""
        try:
            if hasattr(self.best_pipeline, 'named_steps'):
                # sklearn pipeline
                model = self.best_pipeline.named_steps.get('model')
                if model is None:
                    model = self.best_pipeline.named_steps.get('classifier') or self.best_pipeline.named_steps.get('regressor')
                
                if model:
                    return type(model).__name__
            else:
                # Direct model
                return type(self.best_pipeline).__name__
        except:
            return "unknown"
    
    def _get_pattern_optimization_suggestions(self) -> Dict[str, Any]:
        """Get optimization suggestions based on learned patterns"""
        suggestions = {}
        
        if not self.intelligent_recommendations:
            return suggestions
        
        # Model recommendations
        model_recs = self.intelligent_recommendations.get("models", [])
        if model_recs:
            high_confidence_models = [
                rec for rec in model_recs 
                if rec.get("priority") == "high" and rec.get("confidence", 0) > 0.8
            ]
            
            if high_confidence_models:
                suggestions["preferred_models"] = [rec["model"] for rec in high_confidence_models]
                suggestions["model_reasoning"] = f"Prioritized {len(high_confidence_models)} high-confidence models based on learned patterns"
        
        # Feature engineering recommendations
        feature_recs = self.intelligent_recommendations.get("feature_engineering", [])
        for rec in feature_recs:
            if rec.get("action") == "avoid":
                feature_type = rec.get("feature", "unknown")
                if feature_type == "TF-IDF":
                    suggestions["avoid_tfidf"] = True
                    suggestions["tfidf_reasoning"] = "Avoiding TF-IDF for small datasets based on learned patterns"
                elif feature_type == "Polynomial Features":
                    suggestions["avoid_polynomial"] = True
                    suggestions["polynomial_reasoning"] = "Avoiding polynomial features for large datasets based on learned patterns"
        
        # Optimization recommendations
        opt_recs = self.intelligent_recommendations.get("optimization", [])
        for rec in opt_recs:
            if rec.get("strategy") == "Adaptive Optimization":
                suggestions["use_adaptive"] = True
                suggestions["adaptive_reasoning"] = "Using adaptive optimization based on learned success patterns"
            elif "suggested_trials" in rec:
                suggestions["trial_limit"] = rec["suggested_trials"]
                suggestions["trial_reasoning"] = f"Using {rec['suggested_trials']} trials based on learned patterns"
        
        return suggestions
    
    def _optimize_search_space_with_patterns(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize search space based on learned patterns"""
        optimized_space = {
            "original_space": "default",
            "optimized_space": {},
            "modifications": [],
            "confidence": recommendations.get("confidence", {})
        }
        
        # Model recommendations
        model_recs = recommendations.get("models", [])
        if model_recs:
            # Prioritize high-confidence models
            high_confidence_models = [
                rec for rec in model_recs 
                if rec.get("priority") == "high" and rec.get("confidence", 0) > 0.8
            ]
            
            if high_confidence_models:
                optimized_space["optimized_space"]["preferred_models"] = [
                    rec["model"] for rec in high_confidence_models
                ]
                optimized_space["modifications"].append(
                    f"Prioritized {len(high_confidence_models)} high-confidence models"
                )
        
        # Feature engineering recommendations
        feature_recs = recommendations.get("feature_engineering", [])
        for rec in feature_recs:
            if rec.get("action") == "avoid":
                # Avoid certain features for this dataset type
                feature_type = rec.get("feature", "unknown")
                if feature_type == "TF-IDF":
                    optimized_space["optimized_space"]["avoid_tfidf"] = True
                    optimized_space["modifications"].append("Disabled TF-IDF for small dataset")
                elif feature_type == "Polynomial Features":
                    optimized_space["optimized_space"]["avoid_polynomial"] = True
                    optimized_space["modifications"].append("Disabled polynomial features for large dataset")
        
        # Optimization recommendations
        opt_recs = recommendations.get("optimization", [])
        for rec in opt_recs:
            if rec.get("strategy") == "Adaptive Optimization":
                optimized_space["optimized_space"]["use_adaptive"] = True
                optimized_space["modifications"].append("Enabled adaptive optimization")
            elif "suggested_trials" in rec:
                optimized_space["optimized_space"]["trial_limit"] = rec["suggested_trials"]
                optimized_space["modifications"].append(f"Limited trials to {rec['suggested_trials']}")
        
        return optimized_space
    
    def _generate_overall_assessment(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment of the AutoML system"""
        assessment = {
            "system_health": "good",
            "pattern_learning_status": "active" if self.enable_pattern_learning else "disabled",
            "actionable_insights_status": "active" if self.enable_actionable_insights else "disabled",
            "key_findings": [],
            "recommendations": [],
            "confidence_level": "medium"
        }
        
        # Assess actionable insights
        actionable_insights = analysis_results.get("actionable_insights", {})
        if actionable_insights and "action_priority" in actionable_insights:
            priorities = actionable_insights["action_priority"]
            high_priority_count = priorities.get("high", 0)
            
            if high_priority_count > 5:
                assessment["system_health"] = "needs_attention"
                assessment["key_findings"].append(f"High number of priority issues: {high_priority_count}")
                assessment["recommendations"].append("Address high-priority issues immediately")
            elif high_priority_count > 2:
                assessment["system_health"] = "fair"
                assessment["key_findings"].append(f"Moderate number of priority issues: {high_priority_count}")
                assessment["recommendations"].append("Review and address priority issues")
        
        # Assess pattern learning
        if self.enable_pattern_learning:
            pattern_recs = analysis_results.get("pattern_recommendations", {})
            if pattern_recs and "models" in pattern_recs:
                model_recs = pattern_recs["models"]
                high_confidence_count = len([
                    rec for rec in model_recs 
                    if rec.get("priority") == "high" and rec.get("confidence", 0) > 0.8
                ])
                
                if high_confidence_count > 3:
                    assessment["confidence_level"] = "high"
                    assessment["key_findings"].append(f"Strong pattern learning: {high_confidence_count} high-confidence model recommendations")
                elif high_confidence_count > 0:
                    assessment["confidence_level"] = "medium"
                    assessment["key_findings"].append(f"Moderate pattern learning: {high_confidence_count} high-confidence model recommendations")
        
        # Assess benchmarking
        benchmarking = analysis_results.get("benchmarking", {})
        if benchmarking and "system_performance" in benchmarking:
            autoforge_performance = benchmarking["system_performance"].get("AutoForge", {})
            if autoforge_performance:
                avg_accuracy = autoforge_performance.get("avg_performance", {}).get("accuracy", 0)
                success_rate = autoforge_performance.get("success_rate", 0)
                
                if avg_accuracy > 0.85 and success_rate > 0.9:
                    assessment["key_findings"].append("AutoForge shows strong performance in benchmarking")
                    assessment["confidence_level"] = "high"
                elif avg_accuracy > 0.75:
                    assessment["key_findings"].append("AutoForge shows good performance in benchmarking")
        
        return assessment
    
    def get_actionable_explanations(self, X, y):
        """Get actionable explanations"""
        try:
            if hasattr(self, 'explainer') and self.explainer:
                explanations = self.explainer.explain(self.best_pipeline, X, y)
                actionable = ActionableExplainability()
                actionable_insights = actionable.generate_actionable_insights(explanations, X, y)
                return actionable_insights
            else:
                return {"error": "No explainer available"}
        except Exception as e:
            logger.error(f"Failed to generate actionable explanations: {e}")
            return {"error": str(e)}
    
    def get_actionable_summary(self):
        """Get actionable summary"""
        try:
            if hasattr(self, '_last_actionable_insights'):
                actionable = ActionableExplainability()
                return actionable.get_actionable_summary(self._last_actionable_insights)
            else:
                return "No actionable insights available"
        except Exception as e:
            logger.error(f"Failed to generate actionable summary: {e}")
            return f"Error: {e}"
    
    def get_intelligent_recommendations(self, dataset_info):
        """Get intelligent recommendations based on patterns"""
        try:
            if hasattr(self, 'pattern_learner'):
                return self.pattern_learner.get_recommendations(dataset_info, self.task_type)
            else:
                return {"models": [], "feature_engineering": [], "preprocessing": []}
        except Exception as e:
            logger.error(f"Failed to get intelligent recommendations: {e}")
            return {"error": str(e)}
    
    def get_comprehensive_report(self) -> str:
        """
        Generate comprehensive report of all self-improving features
        """
        try:
            report_lines = []
            
            report_lines.append("🧠 SELF-IMPROVING AUTOML REPORT")
            report_lines.append("=" * 60)
            
            # Pattern learning status
            report_lines.append(f"\n🔍 Pattern Learning: {'✅ Active' if self.enable_pattern_learning else '❌ Disabled'}")
            if self.enable_pattern_learning and self.learned_patterns:
                report_lines.append(f"   Learned Patterns: {len(self.learned_patterns)} experiment results stored")
            
            # Intelligent recommendations
            if self.intelligent_recommendations:
                report_lines.append(f"\n🧠 Intelligent Recommendations:")
                
                model_recs = self.intelligent_recommendations.get("models", [])
                if model_recs:
                    report_lines.append(f"   Model Recommendations: {len(model_recs)}")
                    for rec in model_recs[:3]:  # Top 3
                        confidence = rec.get("confidence", 0)
                        priority = rec.get("priority", "medium")
                        report_lines.append(f"     • {rec.get('model', 'unknown')}: {priority} priority, {confidence:.2f} confidence")
                
                feature_recs = self.intelligent_recommendations.get("feature_engineering", [])
                if feature_recs:
                    report_lines.append(f"   Feature Engineering: {len(feature_recs)} recommendations")
                
                opt_recs = self.intelligent_recommendations.get("optimization", [])
                if opt_recs:
                    report_lines.append(f"   Optimization: {len(opt_recs)} recommendations")
            
            # Actionable insights status
            report_lines.append(f"\n💡 Actionable Insights: {'✅ Active' if self.enable_actionable_insights else '❌ Disabled'}")
            
            # Overall assessment
            report_lines.append(f"\n📊 System Assessment:")
            report_lines.append(f"   Pattern Learning: {'Enabled' if self.enable_pattern_learning else 'Disabled'}")
            report_lines.append(f"   Actionable Insights: {'Enabled' if self.enable_actionable_insights else 'Disabled'}")
            report_lines.append(f"   Self-Improving: {'Active' if self.enable_pattern_learning or self.enable_actionable_insights else 'Standard'}")
            
            report_lines.append(f"\n🎯 Key Benefits:")
            if self.enable_pattern_learning:
                report_lines.append("   • Learns from past experiments to improve future performance")
                report_lines.append("   • Provides intelligent recommendations based on learned patterns")
                report_lines.append("   • Optimizes search space dynamically")
            
            if self.enable_actionable_insights:
                report_lines.append("   • Generates actionable insights from model explanations")
                report_lines.append("   • Identifies data quality issues and model risks")
                report_lines.append("   • Provides specific recommendations for improvement")
            
            report_lines.append(f"\n📈 Next Steps:")
            if self.enable_pattern_learning:
                report_lines.append("   1. Store experiment results to build pattern knowledge")
                report_lines.append("   2. Use intelligent recommendations for model selection")
                report_lines.append("   3. Apply pattern-based optimizations to search space")
            
            if self.enable_actionable_insights:
                report_lines.append("   4. Review actionable insights for data improvements")
                report_lines.append("   5. Address high-priority recommendations first")
                report_lines.append("   6. Use actionable explanations for business decisions")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
            return f"Error generating report: {e}"


# Convenience functions
def create_self_improving_automl(**kwargs) -> SelfImprovingAutoML:
    """Convenience function for creating self-improving AutoML"""
    return SelfImprovingAutoML(**kwargs)


def run_comprehensive_analysis(X, y) -> Dict[str, Any]:
    """Convenience function for comprehensive analysis"""
    automl = SelfImprovingAutoML(
        enable_pattern_learning=True,
        enable_actionable_insights=True
    )
    return automl.run_comprehensive_analysis(X, y)
