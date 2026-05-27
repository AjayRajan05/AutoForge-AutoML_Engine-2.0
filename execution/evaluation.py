"""
📊 Evaluation System - Comprehensive model evaluation
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
    confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation system
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.evaluation_history = []
        self.metrics_cache = {}
        
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                      task_type: str = 'classification',
                      X_train: Optional[pd.DataFrame] = None,
                      y_train: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            task_type: 'classification' or 'regression'
            X_train: Training features (optional)
            y_train: Training targets (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            logger.info(f"📊 Evaluating {task_type} model...")
            
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Task-specific evaluation
            if task_type == 'classification':
                results = self._evaluate_classification(
                    model, X_test, y_test, y_pred, X_train, y_train
                )
            else:
                results = self._evaluate_regression(
                    model, X_test, y_test, y_pred, X_train, y_train
                )
            
            # Add general information
            results.update({
                'task_type': task_type,
                'test_samples': len(X_test),
                'test_features': X_test.shape[1],
                'model_type': type(model).__name__,
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            })
            
            # Store evaluation history
            self.evaluation_history.append(results)
            
            logger.info(f"✅ Evaluation complete: {results['primary_score']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            return self._get_fallback_evaluation(task_type)
    
    def _evaluate_classification(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                               y_pred: pd.Series, X_train: Optional[pd.DataFrame],
                               y_train: Optional[pd.Series]) -> Dict[str, Any]:
        """Evaluate classification model"""
        results = {}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_test, y_pred)
        results['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        results['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        results['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC (binary classification)
        if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                results['roc_auc'] = roc_auc_score(y_test, y_proba)
            except Exception as e:
                logger.warning(f"ROC AUC calculation failed: {e}")
                results['roc_auc'] = None
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        # Classification report
        try:
            results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        except Exception as e:
            logger.warning(f"Classification report failed: {e}")
            results['classification_report'] = None
        
        # Primary score for comparison
        results['primary_score'] = results['accuracy']
        
        # Additional metrics
        if X_train is not None and y_train is not None:
            # Training performance
            y_train_pred = model.predict(X_train)
            results['train_accuracy'] = accuracy_score(y_train, y_train_pred)
            
            # Overfitting detection
            overfitting_ratio = results['train_accuracy'] - results['accuracy']
            results['overfitting_ratio'] = overfitting_ratio
            results['is_overfitting'] = overfitting_ratio > 0.1
        
        return results
    
    def _evaluate_regression(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                           y_pred: pd.Series, X_train: Optional[pd.DataFrame],
                           y_train: Optional[pd.Series]) -> Dict[str, Any]:
        """Evaluate regression model"""
        results = {}
        
        # Basic metrics
        results['mse'] = mean_squared_error(y_test, y_pred)
        results['rmse'] = np.sqrt(results['mse'])
        results['mae'] = mean_absolute_error(y_test, y_pred)
        results['r2_score'] = r2_score(y_test, y_pred)
        results['explained_variance'] = explained_variance_score(y_test, y_pred)
        
        # Primary score for comparison (using R2)
        results['primary_score'] = results['r2_score']
        
        # Additional metrics
        results['mean_absolute_percentage_error'] = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        results['max_error'] = np.max(np.abs(y_test - y_pred))
        
        # Residual analysis
        residuals = y_test - y_pred
        results['residual_mean'] = np.mean(residuals)
        results['residual_std'] = np.std(residuals)
        
        # Training performance
        if X_train is not None and y_train is not None:
            y_train_pred = model.predict(X_train)
            train_mse = mean_squared_error(y_train, y_train_pred)
            results['train_mse'] = train_mse
            results['train_rmse'] = np.sqrt(train_mse)
            
            # Overfitting detection
            overfitting_ratio = results['train_mse'] - results['mse']
            results['overfitting_ratio'] = overfitting_ratio
            results['is_overfitting'] = overfitting_ratio > 0.1
        
        return results
    
    def compare_models(self, models: List[Any], X_test: pd.DataFrame, y_test: pd.Series,
                      model_names: List[str], task_type: str = 'classification') -> Dict[str, Any]:
        """Compare multiple models"""
        try:
            logger.info(f"📊 Comparing {len(models)} models...")
            
            comparison_results = {
                'model_names': model_names,
                'task_type': task_type,
                'model_scores': {},
                'ranking': [],
                'best_model': None,
                'best_score': -np.inf if task_type == 'classification' else -np.inf
            }
            
            for i, (model, name) in enumerate(zip(models, model_names)):
                results = self.evaluate_model(model, X_test, y_test, task_type)
                comparison_results['model_scores'][name] = results
                
                # Update best model
                if task_type == 'classification':
                    if results['primary_score'] > comparison_results['best_score']:
                        comparison_results['best_score'] = results['primary_score']
                        comparison_results['best_model'] = name
                else:  # regression
                    if results['primary_score'] > comparison_results['best_score']:
                        comparison_results['best_score'] = results['primary_score']
                        comparison_results['best_model'] = name
            
            # Create ranking
            scored_models = [(name, results['primary_score']) 
                           for name, results in comparison_results['model_scores'].items()]
            
            if task_type == 'classification':
                scored_models.sort(key=lambda x: x[1], reverse=True)  # Higher is better
            else:
                scored_models.sort(key=lambda x: x[1], reverse=True)  # Higher R2 is better
            
            comparison_results['ranking'] = scored_models
            
            logger.info(f"✅ Model comparison complete. Best: {comparison_results['best_model']}")
            return comparison_results
            
        except Exception as e:
            logger.error(f"❌ Model comparison failed: {e}")
            return {'error': str(e)}
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from model"""
        try:
            importance = {}
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                # Linear models
                coef = model.coef_
                if len(coef.shape) > 1:
                    # Multi-class
                    coef = np.mean(np.abs(coef), axis=0)
                importance = dict(zip(feature_names, np.abs(coef)))
            
            # Sort by importance
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            logger.warning(f"Feature importance extraction failed: {e}")
            return {}
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable evaluation report"""
        lines = []
        
        # Header
        lines.append("📊 Model Evaluation Report")
        lines.append("=" * 40)
        
        # Basic info
        lines.append(f"Model Type: {results.get('model_type', 'Unknown')}")
        lines.append(f"Task Type: {results.get('task_type', 'Unknown')}")
        lines.append(f"Test Samples: {results.get('test_samples', 0)}")
        lines.append(f"Features: {results.get('test_features', 0)}")
        
        # Primary score
        lines.append(f"\n🎯 Primary Score: {results.get('primary_score', 'N/A'):.4f}")
        
        # Task-specific metrics
        if results.get('task_type') == 'classification':
            lines.append("\n📈 Classification Metrics:")
            lines.append(f"  Accuracy: {results.get('accuracy', 'N/A'):.4f}")
            lines.append(f"  Precision: {results.get('precision', 'N/A'):.4f}")
            lines.append(f"  Recall: {results.get('recall', 'N/A'):.4f}")
            lines.append(f"  F1 Score: {results.get('f1_score', 'N/A'):.4f}")
            
            if results.get('roc_auc') is not None:
                lines.append(f"  ROC AUC: {results.get('roc_auc', 'N/A'):.4f}")
            
            if results.get('is_overfitting'):
                lines.append(f"  ⚠️  Overfitting detected (ratio: {results.get('overfitting_ratio', 0):.4f})")
        
        else:  # regression
            lines.append("\n📈 Regression Metrics:")
            lines.append(f"  R² Score: {results.get('r2_score', 'N/A'):.4f}")
            lines.append(f"  MSE: {results.get('mse', 'N/A'):.4f}")
            lines.append(f"  RMSE: {results.get('rmse', 'N/A'):.4f}")
            lines.append(f"  MAE: {results.get('mae', 'N/A'):.4f}")
            lines.append(f"  MAPE: {results.get('mean_absolute_percentage_error', 'N/A'):.2f}%")
            
            if results.get('is_overfitting'):
                lines.append(f"  ⚠️  Overfitting detected (ratio: {results.get('overfitting_ratio', 0):.4f})")
        
        return "\n".join(lines)
    
    def _get_fallback_evaluation(self, task_type: str) -> Dict[str, Any]:
        """Get fallback evaluation results"""
        return {
            'task_type': task_type,
            'primary_score': 0.0,
            'error': 'Evaluation failed',
            'model_type': 'Unknown',
            'test_samples': 0,
            'test_features': 0
        }
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations"""
        if not self.evaluation_history:
            return {'message': 'No evaluation history'}
        
        summary = {
            'total_evaluations': len(self.evaluation_history),
            'task_types': list(set(eval['task_type'] for eval in self.evaluation_history)),
            'latest_evaluation': self.evaluation_history[-1],
            'average_scores': {}
        }
        
        # Calculate average scores by task type
        for task_type in summary['task_types']:
            task_evaluations = [eval for eval in self.evaluation_history if eval['task_type'] == task_type]
            scores = [eval['primary_score'] for eval in task_evaluations if 'primary_score' in eval]
            if scores:
                summary['average_scores'][task_type] = np.mean(scores)
        
        return summary
