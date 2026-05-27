"""
⚡ Execution Engine - Model training and optimization execution
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error, f1_score, r2_score,
)
import bootstrap  # noqa: F401
from registry.model_registry import model_registry
from registry.feature_registry import feature_registry
from execution.model_search import ModelSearch
from execution.preprocessing_pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Model training and optimization execution engine
    
    Handles model training, evaluation, and optimization based on
    strategy from the intelligence engine.
    """
    
    def __init__(self):
        """Initialize execution engine"""
        self.model_registry = model_registry
        self.feature_registry = feature_registry
        self.training_history = []
        self.current_models = {}
        self.performance_cache = {}
        self.model_search = ModelSearch()
        
    def run(self, features: pd.DataFrame, target: pd.Series, 
            config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute model training and optimization
        
        Args:
            features: Feature data
            target: Target data
            config: Execution configuration
            
        Returns:
            Dictionary with execution results
        """
        try:
            logger.info("⚡ Starting model execution...")
            start_time = time.time()
            
            # Step 1: Data preprocessing (skip if unified layer already preprocessed)
            if config.get('skip_preprocessing') and config.get('preprocessing_artifacts'):
                logger.info("🔧 Step 1: Using preprocessed data from pipeline...")
                X_processed = features.copy()
                y_processed = target.copy()
                preprocessing_info = config['preprocessing_artifacts']
            else:
                logger.info("🔧 Step 1: Data preprocessing...")
                X_processed, y_processed, preprocessing_info = self._preprocess_data(
                    features, target, config
                )

            if config.get('X_holdout') is not None:
                config['X_holdout'] = self._transform_holdout(
                    config['X_holdout'], preprocessing_info, config
                )
            
            # Step 2: Advanced feature processing (using feature registry)
            logger.info("🧩 Step 2: Advanced feature processing...")
            X_enhanced, y_enhanced, feature_processing_info = self._apply_advanced_features(
                X_processed, y_processed, config
            )
            if isinstance(X_enhanced, pd.Series):
                X_enhanced = X_enhanced.to_frame()
            elif not isinstance(X_enhanced, pd.DataFrame):
                X_enhanced = pd.DataFrame(X_enhanced)
            
            # Step 3: Model training and evaluation
            logger.info("🤖 Step 3: Model training and evaluation...")
            model_results = self._train_models_by_family(
                X_enhanced, y_processed, config
            )
            
            # Step 4: Model evaluation and selection (CV on train; holdout when provided)
            logger.info("📊 Step 4: Model evaluation and selection...")
            best_model, best_score, evaluation_results = self._evaluate_models(
                model_results, X_enhanced, y_processed, config
            )
            holdout_metrics = evaluation_results.get('holdout_metrics', {})
            
            # Step 5: Final training on full dataset
            logger.info("🎯 Step 5: Final training on full dataset...")
            final_model = self._train_final_model(
                best_model, X_enhanced, y_processed, config
            )
            
            # Step 6: Compile results
            execution_time = time.time() - start_time
            if isinstance(X_enhanced, pd.Series):
                X_enhanced = X_enhanced.to_frame()
            elif not isinstance(X_enhanced, pd.DataFrame):
                X_enhanced = pd.DataFrame(X_enhanced)

            results = {
                'best_model': final_model,
                'best_score': best_score,
                'model_type': type(final_model).__name__,
                'training_time': execution_time,
                'preprocessing_info': preprocessing_info,
                'feature_processing_info': feature_processing_info,
                'model_results': model_results,
                'evaluation_results': evaluation_results,
                'X_train': X_enhanced,
                'y_train': y_processed,
                'feature_names': X_enhanced.columns.tolist(),
                'n_features': len(X_enhanced.columns),
                'n_samples': len(X_enhanced),
                'task_type': config.get('task_type', 'classification'),
                'X_holdout': config.get('X_holdout'),
                'y_holdout': config.get('y_holdout'),
                'holdout_metrics': holdout_metrics,
                'metadata': {
                    'execution_timestamp': time.time(),
                    'models_tried': list(model_results.keys()),
                    'cross_validation_folds': config.get('cv_folds', 5),
                    'optimization_trials': config.get('max_trials', 50),
                    'advanced_features_used': feature_processing_info.get('features_applied', []),
                    'model_family': config.get('model_family', 'ml'),
                    'holdout_fraction': config.get('holdout_fraction', 0.2),
                    'scoring': config.get('scoring'),
                }
            }
            
            # Store in cache
            self.performance_cache[config.get('task_type', 'unknown')] = results
            self.training_history.append(results)
            
            logger.info(f"✅ Model execution complete: {best_score:.4f} score in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ Model execution failed: {e}")
            raise
    
    def _transform_holdout(
        self,
        X_holdout: pd.DataFrame,
        preprocessing_info: Dict[str, Any],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Apply the same preprocessing artifacts to holdout rows."""
        if config.get('skip_preprocessing') and config.get('preprocessing_artifacts'):
            return X_holdout
        artifacts = config.get('preprocessing_artifacts') or preprocessing_info or {}
        if not artifacts:
            return X_holdout
        try:
            pipeline = PreprocessingPipeline()
            pipeline.artifacts = artifacts
            pipeline.is_fitted = True
            pipeline.column_transformer = artifacts.get('column_transformer')
            return pipeline.transform(X_holdout)
        except Exception as exc:
            logger.warning("Holdout transform failed, using raw features: %s", exc)
            return X_holdout

    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series,
                        config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Preprocess via unified PreprocessingPipeline (ColumnTransformer artifacts)."""
        try:
            task_type = config.get('task_type', 'classification')
            pipeline = PreprocessingPipeline(
                scale_features=config.get('scale_features', True),
                feature_engineering=config.get('feature_engineering', False),
                recipe_name=config.get('preprocessing_recipe'),
            )
            if config.get('preprocessing_recipe_config'):
                from execution.preprocessing_pipeline import PreprocessingPipeline as PP
                pipeline = PP.from_recipe(
                    config['preprocessing_recipe'],
                    config.get('preprocessing_recipe_config'),
                )
            return pipeline.fit_transform(X, y, task_type=task_type)
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            return X, y, {'error': str(e)}
    
    def _apply_advanced_features(self, X: pd.DataFrame, y: pd.Series, 
                                config: Dict[str, Any]) -> tuple:
        """Apply advanced feature processing using feature registry"""
        try:
            # Get available features
            available_features = self.feature_registry.list_features()
            feature_config = config.get('advanced_features', {})
            
            # Determine which features to apply based on data characteristics
            features_to_apply = []
            
            # NAS for neural architecture search (if appropriate)
            if feature_config.get('use_nas', False) and 'nas' in available_features:
                min_features = feature_config.get('nas_min_features', 10)
                min_samples = feature_config.get('nas_min_samples', 100)
                if X.shape[1] >= min_features and X.shape[0] >= min_samples:
                    features_to_apply.append('nas')
            
            # Multimodal for mixed data types
            if feature_config.get('use_multimodal', False) and 'multimodal' in available_features:
                if len(X.select_dtypes(include=['object', 'category']).columns) > 0:
                    features_to_apply.append('multimodal')
            
            # Distributed for large datasets
            if feature_config.get('use_distributed', False) and 'distributed' in available_features:
                if X.shape[0] > 10000:  # Large dataset
                    features_to_apply.append('distributed')
            
            # Meta-learning for strategy enhancement
            if feature_config.get('use_meta_learning', False) and 'meta_learning' in available_features:
                features_to_apply.append('meta_learning')
            
            # Explainability for feature importance
            if feature_config.get('use_explainability', False) and 'explainability' in available_features:
                features_to_apply.append('explainability')
            
            # Bulletproof for robustness
            if feature_config.get('use_bulletproof', False) and 'bulletproof' in available_features:
                features_to_apply.append('bulletproof')
            
            # Apply feature pipeline
            if features_to_apply:
                logger.info(f"🧩 Applying advanced features: {features_to_apply}")
                
                feature_pipeline = self.feature_registry.create_feature_pipeline(features_to_apply)
                X_enhanced, y_enhanced = feature_pipeline.execute(X, y)
                
                # Get execution summary
                pipeline_summary = feature_pipeline.get_execution_summary()
                
                feature_info = {
                    'features_applied': features_to_apply,
                    'pipeline_summary': pipeline_summary,
                    'original_shape': X.shape,
                    'enhanced_shape': X_enhanced.shape,
                    'success': pipeline_summary['successful'] > 0
                }
                
                logger.info(f"✅ Advanced features applied: {pipeline_summary['successful']}/{len(features_to_apply)} successful")
                
            else:
                # No advanced features applied
                X_enhanced = X.copy()
                y_enhanced = y.copy() if y is not None else y
                
                feature_info = {
                    'features_applied': [],
                    'pipeline_summary': None,
                    'original_shape': X.shape,
                    'enhanced_shape': X_enhanced.shape,
                    'success': True,
                    'message': 'No advanced features applied'
                }
            
            return X_enhanced, y_enhanced, feature_info
            
        except Exception as e:
            logger.warning(f"⚠️ Advanced feature processing failed: {e}")
            # Return original data as fallback
            return X, y, {'error': str(e), 'success': False}

    def _train_models_by_family(self, X: pd.DataFrame, y: pd.Series,
                                config: Dict[str, Any]) -> Dict[str, Any]:
        """Train models based on model_family setting (ml, dl, or both)."""
        model_family = config.get('model_family', 'ml')
        combined: Dict[str, Any] = {}

        if model_family in ('ml', 'both'):
            combined.update(self._train_models(X, y, config))

        if model_family in ('dl', 'both'):
            try:
                from execution.dl_trainer import DLTrainer, DL_AVAILABLE, require_dl_dependencies
                if not DL_AVAILABLE:
                    require_dl_dependencies()
                dl_trainer = DLTrainer()
                dl_results = dl_trainer.train(
                    X,
                    y,
                    task_type=config.get('task_type', 'classification'),
                    max_trials=config.get('dl_max_trials', 5),
                    epochs=config.get('dl_epochs', 10),
                    random_state=config.get('random_state', 42),
                )
                combined.update(dl_results)
            except ImportError as exc:
                if model_family == 'dl':
                    raise
                logger.warning("DL training skipped (optional deps missing): %s", exc)

        if not combined:
            combined = self._train_models(X, y, config)
        return combined
    
    def _train_models(self, X: pd.DataFrame, y: pd.Series, 
                     config: Dict[str, Any]) -> Dict[str, Any]:
        """Train multiple models based on strategy"""
        task_type = config.get('task_type', 'classification')
        
        # Ensure X is a DataFrame, not a Series
        if isinstance(X, pd.Series):
            X = X.to_frame()
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        models_to_try = config.get('models', self._get_default_models(task_type, X))
        
        model_results = {}
        cv_folds = config.get('cv_folds', 5)
        n_jobs = config.get('n_jobs', -1)
        search_depth = config.get('search_depth', 'balanced')
        budget = config.get('search_budget') or {}
        scoring = config.get('scoring')
        if scoring is None:
            from execution.preprocessing_search import default_scoring
            scoring = default_scoring(task_type)
        evaluate_variants = config.get(
            'evaluate_preprocessing_variants',
            False,
        )
        best_cv_so_far = float('-inf')
        early_stop = config.get('early_stop_models', False)

        for model_name in models_to_try:
            try:
                logger.info(f"🤖 Training {model_name}...")
                
                def model_factory(**params):
                    return self.model_registry.get_model(
                        model_name, task_type, **params
                    )

                result = self.model_search.search_model(
                    model_name=model_name,
                    model_factory=model_factory,
                    X=X,
                    y=y,
                    task_type=task_type,
                    search_depth=search_depth,
                    cv_folds=cv_folds,
                    n_jobs=n_jobs,
                    max_trials=budget.get('max_trials_per_model'),
                    timeout_per_model=budget.get('timeout_per_model'),
                    evaluate_variants=evaluate_variants,
                    scoring=scoring,
                    best_cv_so_far=best_cv_so_far if early_stop else None,
                    early_stop_margin=config.get('early_stop_margin', 0.05),
                )
                X_holdout = config.get('X_holdout')
                y_holdout = config.get('y_holdout')
                if (
                    X_holdout is not None
                    and y_holdout is not None
                    and 'error' not in result
                    and result.get('model') is not None
                ):
                    try:
                        model = result['model']
                        if not hasattr(model, 'predict'):
                            model.fit(X, y)
                        y_pred = model.predict(X_holdout)
                        holdout = self._compute_holdout_metrics(y_holdout, y_pred, task_type)
                        result['holdout_score'] = holdout.get('primary_score')
                        result['holdout_metrics'] = holdout
                    except Exception as exc:
                        logger.debug("Holdout eval failed for %s: %s", model_name, exc)
                model_results[model_name] = result
                if 'cv_mean' in result and 'error' not in result:
                    best_cv_so_far = max(best_cv_so_far, result['cv_mean'])
                logger.info(
                    f"✅ {model_name}: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}"
                )
                
            except Exception as e:
                logger.warning(f"❌ {model_name} failed: {e}")
                model_results[model_name] = {
                    'error': str(e),
                    'cv_mean': 0.0,
                    'cv_std': 0.0
                }
        
        return model_results
    
    def _evaluate_models(self, model_results: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                        config: Dict[str, Any]) -> Tuple[Any, float, Dict[str, Any]]:
        """Evaluate models and select the best one"""
        task_type = config.get('task_type', 'classification')
        
        # Filter out failed models
        valid_models = {
            name: result for name, result in model_results.items()
            if 'error' not in result
        }
        
        if not valid_models:
            raise ValueError("No models trained successfully")
        
        # Select best model
        if task_type == 'classification':
            best_name = max(valid_models.keys(), key=lambda x: valid_models[x]['cv_mean'])
        else:
            # For regression, higher (less negative) MSE is better
            best_name = max(valid_models.keys(), key=lambda x: valid_models[x]['cv_mean'])
        
        best_result = valid_models[best_name]
        best_model = best_result['model']
        best_score = best_result['cv_mean']
        
        evaluation_results = {}
        X_holdout = config.get('X_holdout')
        y_holdout = config.get('y_holdout')

        if X_holdout is not None and y_holdout is not None:
            best_model.fit(X, y)
            y_pred = best_model.predict(X_holdout)
            holdout_metrics = self._compute_holdout_metrics(
                y_holdout, y_pred, task_type
            )
            evaluation_results['holdout_metrics'] = holdout_metrics
            evaluation_results.update(holdout_metrics)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=config.get('random_state', 42),
                stratify=y if task_type == 'classification' else None,
            )
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            holdout_metrics = self._compute_holdout_metrics(y_test, y_pred, task_type)
            evaluation_results['holdout_metrics'] = holdout_metrics
            evaluation_results.update(holdout_metrics)
        
        # Add feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            elif isinstance(X, pd.Series):
                feature_names = [X.name or 'feature_0']
            else:
                feature_names = [f'feature_{i}' for i in range(len(best_model.feature_importances_))]
            evaluation_results['feature_importance'] = dict(
                zip(feature_names, best_model.feature_importances_)
            )
        
        evaluation_results['all_model_scores'] = {
            name: {'mean': result['cv_mean'], 'std': result['cv_std']}
            for name, result in valid_models.items()
        }
        
        return best_model, best_score, evaluation_results

    @staticmethod
    def _compute_holdout_metrics(
        y_true: pd.Series, y_pred: np.ndarray, task_type: str
    ) -> Dict[str, float]:
        """Holdout metrics for classification or regression."""
        if task_type == 'classification':
            return {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'f1_score': float(f1_score(y_true, y_pred, average='weighted')),
                'primary_score': float(accuracy_score(y_true, y_pred)),
            }
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            'mse': mse,
            'rmse': float(np.sqrt(mse)),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2_score': float(r2_score(y_true, y_pred)),
            'primary_score': float(r2_score(y_true, y_pred)),
        }
    
    def _train_final_model(self, best_model: Any, X: pd.DataFrame, y: pd.Series,
                          config: Dict[str, Any]) -> Any:
        """Train final model on full dataset"""
        try:
            if hasattr(best_model, 'backend') and best_model.backend == 'keras':
                return best_model
            # Create fresh instance of the best model
            model_type = type(best_model)
            final_model = model_type(**self._get_model_params(model_type, config))
            
            # Train on full dataset
            final_model.fit(X, y)
            
            return final_model
            
        except Exception as e:
            logger.warning(f"Final model training failed, using trained model: {e}")
            return best_model
    
        
    def _get_default_models(self, task_type: str, features: pd.DataFrame) -> List[str]:
        """Get default models for task type"""
        # Handle case where features might be a Series
        if hasattr(features, 'columns'):
            n_features = len(features.columns)
        else:
            n_features = 1  # Single feature case
        
        return self.model_registry.recommend_models(len(features), n_features, task_type)
    
    def _get_model_params(self, model_type: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get model parameters based on configuration"""
        base_params = {
            'random_state': config.get('random_state', 42),
            'n_jobs': config.get('n_jobs', -1)
        }
        
        # Model-specific parameters
        if 'RandomForest' in str(model_type):
            base_params.update({
                'n_estimators': config.get('n_estimators', 100),
                'max_depth': config.get('max_depth', None)
            })
        elif 'LogisticRegression' in str(model_type):
            base_params.update({
                'max_iter': config.get('max_iter', 1000)
            })
        elif 'SVC' in str(model_type) or 'SVR' in str(model_type):
            base_params.update({
                'kernel': 'rbf'
            })
        
        return base_params
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.training_history:
            return {'message': 'No training history available'}
        
        latest = self.training_history[-1]
        
        return {
            'total_executions': len(self.training_history),
            'latest_execution': {
                'best_score': latest['best_score'],
                'training_time': latest['training_time'],
                'model_type': latest['model_type'],
                'n_features': latest['n_features']
            },
            'average_score': np.mean([r['best_score'] for r in self.training_history]),
            'average_time': np.mean([r['training_time'] for r in self.training_history])
        }
