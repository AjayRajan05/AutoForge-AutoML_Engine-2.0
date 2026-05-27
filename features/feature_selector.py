"""
🔧 Feature Selector - Intelligent feature selection based on dataset characteristics
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Intelligent Feature Selector
    
    Selects optimal features based on dataset characteristics,
    constraints, and performance requirements.
    """
    
    def __init__(self):
        """Initialize feature selector"""
        self.selection_methods = {
            'univariate': self._univariate_selection,
            'model_based': self._model_based_selection,
            'correlation': self._correlation_selection,
            'variance': self._variance_selection
        }
        
        self.selected_features = []
        self.feature_scores = {}
        self.selection_metadata = {}
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       profile: Dict[str, Any] = None,
                       config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Select optimal features based on dataset characteristics
        
        Args:
            X: Feature data
            y: Target data
            profile: Dataset profile from intelligence engine
            config: Selection configuration
            
        Returns:
            Dictionary with selection results
        """
        try:
            logger.info("🔧 Starting intelligent feature selection...")
            
            # Determine selection strategy
            strategy = self._determine_selection_strategy(profile, config)
            
            # Apply selection methods
            selected_features = set()
            all_scores = {}
            method_results = {}
            
            for method in strategy['methods']:
                if method in self.selection_methods:
                    logger.info(f"Applying {method} feature selection...")
                    
                    method_features, method_scores = self.selection_methods[method](
                        X, y, profile, config
                    )
                    
                    selected_features.update(method_features)
                    all_scores.update(method_scores)
                    method_results[method] = {
                        'features': method_features,
                        'scores': method_scores,
                        'n_selected': len(method_features)
                    }
            
            # Apply constraints
            final_features = self._apply_constraints(
                list(selected_features), all_scores, profile, config
            )
            
            # Compile results
            results = {
                'selected_features': final_features,
                'n_selected': len(final_features),
                'n_original': X.shape[1],
                'selection_ratio': len(final_features) / X.shape[1],
                'feature_scores': {feat: all_scores.get(feat, 0) for feat in final_features},
                'method_results': method_results,
                'strategy': strategy,
                'selection_metadata': {
                    'dataset_size': profile.get('size_profile', {}).get('n_samples', 0),
                    'task_type': profile.get('complexity_profile', {}).get('task_type', 'unknown'),
                    'data_type': profile.get('data_type', 'unknown'),
                    'selection_timestamp': pd.Timestamp.now().isoformat()
                }
            }
            
            # Store for later use
            self.selected_features = final_features
            self.feature_scores = results['feature_scores']
            self.selection_metadata = results['selection_metadata']
            
            logger.info(f"✅ Feature selection complete: {len(final_features)}/{X.shape[1]} features selected")
            return results
            
        except Exception as e:
            logger.error(f"❌ Feature selection failed: {e}")
            # Return all features as fallback
            return {
                'selected_features': X.columns.tolist(),
                'n_selected': X.shape[1],
                'n_original': X.shape[1],
                'selection_ratio': 1.0,
                'feature_scores': {col: 1.0 for col in X.columns},
                'method_results': {},
                'strategy': {'fallback': True},
                'selection_metadata': {'error': str(e)}
            }
    
    def _determine_selection_strategy(self, profile: Dict[str, Any], 
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal feature selection strategy"""
        size_profile = profile.get('size_profile', {})
        complexity_profile = profile.get('complexity_profile', {})
        data_type = profile.get('data_type', 'tabular')
        
        n_samples = size_profile.get('n_samples', 0)
        n_features = size_profile.get('n_features', 0)
        task_type = complexity_profile.get('task_type', 'classification')
        
        # Base strategy
        strategy = {
            'methods': ['univariate', 'variance'],
            'max_features': config.get('max_features', n_features),
            'min_features': config.get('min_features', max(1, n_features // 10)),
            'selection_pressure': config.get('selection_pressure', 'medium')
        }
        
        # Adjust based on dataset size
        if n_samples < 1000:  # Small dataset
            strategy['methods'].append('model_based')
            strategy['max_features'] = min(strategy['max_features'], n_features // 2)
        
        elif n_samples > 100000:  # Large dataset
            strategy['methods'] = ['variance', 'univariate']
            strategy['max_features'] = min(strategy['max_features'], 1000)
        
        # Adjust based on feature count
        if n_features > 1000:  # High dimensional
            strategy['methods'].insert(0, 'variance')
            strategy['max_features'] = min(strategy['max_features'], 500)
        
        # Adjust based on data type
        if data_type == 'text':
            strategy['max_features'] = min(strategy['max_features'], 10000)
        elif data_type == 'time_series':
            strategy['methods'] = ['variance', 'correlation']
        
        # Adjust based on task complexity
        if complexity_profile.get('is_imbalanced', False):
            strategy['selection_pressure'] = 'high'
        
        return strategy
    
    def _univariate_selection(self, X: pd.DataFrame, y: pd.Series,
                            profile: Dict[str, Any], config: Dict[str, Any]) -> tuple:
        """Univariate statistical feature selection"""
        try:
            task_type = profile.get('complexity_profile', {}).get('task_type', 'classification')
            max_features = config.get('max_features', X.shape[1])
            
            # Select appropriate score function
            if task_type == 'classification':
                score_func = f_classif
            else:
                score_func = f_regression
            
            # Handle categorical variables
            X_numeric = X.select_dtypes(include=[np.number])
            
            if X_numeric.empty:
                return X.columns.tolist(), {col: 1.0 for col in X.columns}
            
            # Apply selection
            selector = SelectKBest(score_func=score_func, k=min(max_features, X_numeric.shape[1]))
            selector.fit(X_numeric, y)
            
            # Get selected features and scores
            selected_mask = selector.get_support()
            selected_features = X_numeric.columns[selected_mask].tolist()
            scores = dict(zip(X_numeric.columns, selector.scores_))
            
            # Add categorical features (always include them)
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            selected_features.extend(categorical_cols.tolist())
            for col in categorical_cols:
                scores[col] = 1.0
            
            return selected_features, scores
            
        except Exception as e:
            logger.warning(f"Univariate selection failed: {e}")
            return X.columns.tolist(), {col: 1.0 for col in X.columns}
    
    def _model_based_selection(self, X: pd.DataFrame, y: pd.Series,
                              profile: Dict[str, Any], config: Dict[str, Any]) -> tuple:
        """Model-based feature selection using Random Forest"""
        try:
            task_type = profile.get('complexity_profile', {}).get('task_type', 'classification')
            max_features = config.get('max_features', X.shape[1])
            
            # Prepare data
            X_processed = self._prepare_data_for_model(X)
            
            # Select appropriate model
            if task_type == 'classification':
                model = RandomForestClassifier(n_estimators=config.get('n_estimators', 50), random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=config.get('n_estimators', 50), random_state=42)
            
            # Fit model
            model.fit(X_processed, y)
            
            # Get feature importance
            importances = model.feature_importances_
            feature_names = X_processed.columns
            
            # Select top features
            n_select = min(max_features, len(feature_names))
            top_indices = np.argsort(importances)[-n_select:]
            selected_features = [feature_names[i] for i in top_indices]
            scores = dict(zip(feature_names, importances))
            
            return selected_features, scores
            
        except Exception as e:
            logger.warning(f"Model-based selection failed: {e}")
            return X.columns.tolist(), {col: 1.0 for col in X.columns}
    
    def _correlation_selection(self, X: pd.DataFrame, y: pd.Series,
                             profile: Dict[str, Any], config: Dict[str, Any]) -> tuple:
        """Correlation-based feature selection"""
        try:
            numeric_cols = X.select_dtypes(include=[np.number])
            
            if numeric_cols.shape[1] < 2:
                return X.columns.tolist(), {col: 1.0 for col in X.columns}
            
            # Calculate correlation matrix
            corr_matrix = numeric_cols.corr().abs()
            
            # Find highly correlated features
            high_corr_pairs = []
            threshold = config.get('correlation_threshold', 0.8)
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > threshold:
                        high_corr_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))
            
            # Remove one feature from each highly correlated pair
            features_to_remove = set()
            for feat1, feat2, corr in high_corr_pairs:
                # Keep the feature with higher correlation with target (if available)
                if feat1 not in features_to_remove and feat2 not in features_to_remove:
                    # Simple heuristic: remove the second feature
                    features_to_remove.add(feat2)
            
            selected_features = [col for col in X.columns if col not in features_to_remove]
            scores = {col: 1.0 for col in selected_features}
            
            return selected_features, scores
            
        except Exception as e:
            logger.warning(f"Correlation selection failed: {e}")
            return X.columns.tolist(), {col: 1.0 for col in X.columns}
    
    def _variance_selection(self, X: pd.DataFrame, y: pd.Series,
                           profile: Dict[str, Any], config: Dict[str, Any]) -> tuple:
        """Variance-based feature selection"""
        try:
            threshold = config.get('variance_threshold', 0.01)
            
            # Calculate variance for numeric columns
            numeric_cols = X.select_dtypes(include=[np.number])
            
            if numeric_cols.empty:
                return X.columns.tolist(), {col: 1.0 for col in X.columns}
            
            variances = numeric_cols.var()
            
            # Select features with sufficient variance
            high_variance_features = variances[variances > threshold].index.tolist()
            
            # Always include categorical features
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            selected_features = high_variance_features + categorical_cols.tolist()
            
            # Calculate scores (normalized variance)
            max_var = variances.max() if len(variances) > 0 else 1.0
            scores = {}
            
            for col in numeric_cols.columns:
                if col in high_variance_features:
                    scores[col] = variances[col] / max_var
                else:
                    scores[col] = 0.0
            
            for col in categorical_cols:
                scores[col] = 1.0
            
            return selected_features, scores
            
        except Exception as e:
            logger.warning(f"Variance selection failed: {e}")
            return X.columns.tolist(), {col: 1.0 for col in X.columns}
    
    def _prepare_data_for_model(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for model-based selection"""
        X_processed = X.copy()
        
        # Handle categorical variables
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            # Simple label encoding
            X_processed[col] = pd.factorize(X_processed[col])[0]
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.mean())
        
        return X_processed
    
    def _apply_constraints(self, features: List[str], scores: Dict[str, float],
                         profile: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
        """Apply constraints to selected features"""
        # Sort by score
        sorted_features = sorted(features, key=lambda x: scores.get(x, 0), reverse=True)
        
        # Apply minimum/maximum constraints
        min_features = config.get('min_features', 1)
        max_features = config.get('max_features', len(features))
        
        # Adjust based on dataset size
        size_profile = profile.get('size_profile', {})
        n_samples = size_profile.get('n_samples', 0)
        
        if n_samples < 100:
            max_features = min(max_features, n_samples // 2)
        elif n_samples > 100000:
            max_features = min(max_features, 1000)
        
        # Apply constraints
        n_final = max(min_features, min(max_features, len(sorted_features)))
        final_features = sorted_features[:n_final]
        
        return final_features
    
    def get_selection_summary(self) -> str:
        """Get human-readable summary of feature selection"""
        if not self.selected_features:
            return "No feature selection performed"
        
        summary_lines = [
            f"🔧 Feature Selection Summary:",
            f"Selected: {len(self.selected_features)} features",
            f"Top features: {self.selected_features[:5]}"
        ]
        
        if self.feature_scores:
            top_features = sorted(
                self.feature_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            summary_lines.append("Top scores:")
            for feat, score in top_features:
                summary_lines.append(f"  {feat}: {score:.4f}")
        
        return "\n".join(summary_lines)
