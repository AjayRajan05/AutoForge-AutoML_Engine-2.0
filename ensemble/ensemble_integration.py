"""
🔗 Ensemble Integration
Connects AutoForge with existing ensemble modules
"""

import logging
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.base import BaseEstimator

try:
    from . import Stacker, Blender
except ImportError:
    Stacker = None
    Blender = None

logger = logging.getLogger(__name__)


class EnsembleIntegrator:
    """
    Integration layer for ensemble methods
    """
    
    def __init__(self):
        """Initialize ensemble integrator"""
        self.available_methods = self._check_available_methods()
        self.ensemble_history = []
        
    def _check_available_methods(self) -> Dict[str, bool]:
        """Check which ensemble methods are available"""
        methods = {
            'stacker': Stacker is not None,
            'blender': Blender is not None,
            'voting': True,  # Always available from sklearn
            'stacking': True  # Always available from sklearn
        }
        
        available_count = sum(methods.values())
        logger.info(f"🔗 Available ensemble methods: {available_count}/{len(methods)}")
        
        return methods
    
    def persist_top_models(
        self,
        top_items: List[Tuple[str, Dict[str, Any]]],
        save_dir: str,
        run_id: str,
    ) -> List[str]:
        """Save top-N models to disk for ensemble loading."""
        paths = []
        base = Path(save_dir) / run_id
        base.mkdir(parents=True, exist_ok=True)
        for rank, (name, result) in enumerate(top_items):
            model_path = base / f"rank_{rank}_{name}.joblib"
            joblib.dump(result['model'], model_path)
            meta = {
                'name': name,
                'rank': rank,
                'cv_mean': result.get('cv_mean'),
                'best_params': result.get('best_params', {}),
            }
            joblib.dump(meta, base / f"rank_{rank}_{name}_meta.joblib")
            paths.append(str(model_path))
        logger.info("Persisted %d models to %s", len(paths), base)
        return paths

    def build_disk_stacking_ensemble(
        self,
        top_items: List[Tuple[str, Dict[str, Any]]],
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        save_dir: str,
        run_id: str,
    ) -> BaseEstimator:
        """Persist top models and build a stacking ensemble from disk-backed estimators."""
        self.persist_top_models(top_items, save_dir, run_id)
        models = [result['model'] for _, result in top_items]
        model_names = [name for name, _ in top_items]
        if task_type == 'classification':
            ensemble = StackingClassifier(
                estimators=list(zip(model_names, models)),
                cv=3,
                final_estimator=self._get_final_estimator(task_type),
            )
        else:
            ensemble = StackingRegressor(
                estimators=list(zip(model_names, models)),
                cv=3,
                final_estimator=self._get_final_estimator(task_type),
            )
        ensemble.fit(X, y)
        run_path = Path(save_dir) / run_id
        ensemble_path = run_path / "ensemble.joblib"
        joblib.dump(ensemble, ensemble_path)
        meta = {
            "run_id": run_id,
            "task_type": task_type,
            "model_names": model_names,
            "ensemble_path": str(ensemble_path),
        }
        joblib.dump(meta, run_path / "ensemble_meta.joblib")
        self.ensemble_history.append({
            'method': 'disk_stacking',
            'run_id': run_id,
            'n_models': len(models),
            'save_dir': str(run_path),
            'ensemble_path': str(ensemble_path),
        })
        return ensemble

    def load_disk_ensemble(self, run_dir: str) -> BaseEstimator:
        """Load a fitted stacking ensemble from disk without retraining base models."""
        run_path = Path(run_dir)
        ensemble_path = run_path / "ensemble.joblib"
        if ensemble_path.exists():
            logger.info("Loading disk ensemble from %s", ensemble_path)
            return joblib.load(ensemble_path)
        return self.load_stacking_from_disk(run_dir, None, None, "classification")

    def load_stacking_from_disk(
        self,
        run_dir: str,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = 'classification',
    ) -> BaseEstimator:
        """Load persisted models and fit stacking ensemble."""
        run_path = Path(run_dir)
        models = []
        names = []
        for model_file in sorted(run_path.glob("rank_*_*.joblib")):
            if model_file.name.endswith('_meta.joblib'):
                continue
            models.append(joblib.load(model_file))
            names.append(model_file.stem)
        return self.create_stacking_ensemble(models, X, y, task_type)

    def create_voting_ensemble(self, models: List[BaseEstimator],
                             task_type: str = 'classification',
                             voting: str = 'soft') -> BaseEstimator:
        """Create voting ensemble from models"""
        try:
            logger.info(f"🔗 Creating voting ensemble for {task_type}...")
            
            # Create model names
            model_names = [f"model_{i}" for i in range(len(models))]
            
            if task_type == 'classification':
                ensemble = VotingClassifier(
                    estimators=list(zip(model_names, models)),
                    voting=voting
                )
            else:
                ensemble = VotingRegressor(
                    estimators=list(zip(model_names, models))
                )
            
            logger.info(f"✅ Voting ensemble created with {len(models)} models")
            return ensemble
            
        except Exception as e:
            logger.error(f"❌ Voting ensemble creation failed: {e}")
            # Return first model as fallback
            return models[0] if models else None
    
    def create_stacking_ensemble(self, models: List[BaseEstimator], 
                                X: pd.DataFrame, y: pd.Series,
                                task_type: str = 'classification') -> BaseEstimator:
        """Create stacking ensemble from models"""
        try:
            logger.info(f"🔗 Creating stacking ensemble for {task_type}...")
            
            # Create model names
            model_names = [f"model_{i}" for i in range(len(models))]
            
            if task_type == 'classification':
                ensemble = StackingClassifier(
                    estimators=list(zip(model_names, models)),
                    cv=3,
                    final_estimator=self._get_final_estimator(task_type)
                )
            else:
                ensemble = StackingRegressor(
                    estimators=list(zip(model_names, models)),
                    cv=3,
                    final_estimator=self._get_final_estimator(task_type)
                )
            
            logger.info(f"✅ Stacking ensemble created with {len(models)} models")
            return ensemble
            
        except Exception as e:
            logger.error(f"❌ Stacking ensemble creation failed: {e}")
            # Return voting ensemble as fallback
            return self.create_voting_ensemble(models, task_type)
    
    def create_custom_ensemble(self, models: List[BaseEstimator], 
                             X: pd.DataFrame, y: pd.Series,
                             method: str = 'stacking') -> BaseEstimator:
        """Create custom ensemble using existing modules"""
        try:
            if method == 'stacking' and self.available_methods.get('stacker', False):
                logger.info("🔗 Using custom Stacker...")
                stacker = Stacker()
                
                # Train individual models first
                trained_models = []
                for model in models:
                    model.fit(X, y)
                    trained_models.append(model)
                
                # Use custom stacker
                ensemble = stacker.stack_models(trained_models, X, y)
                
            elif method == 'blending' and self.available_methods.get('blender', False):
                logger.info("🔗 Using custom Blender...")
                blender = Blender()
                
                # Train individual models first
                trained_models = []
                for model in models:
                    model.fit(X, y)
                    trained_models.append(model)
                
                # Use custom blender
                ensemble = blender.blend_models(trained_models, X, y)
                
            else:
                # Fallback to sklearn stacking
                task_type = 'classification' if len(np.unique(y)) < 20 else 'regression'
                ensemble = self.create_stacking_ensemble(models, X, y, task_type)
            
            return ensemble
            
        except Exception as e:
            logger.error(f"❌ Custom ensemble creation failed: {e}")
            # Return voting ensemble as fallback
            task_type = 'classification' if len(np.unique(y)) < 20 else 'regression'
            return self.create_voting_ensemble(models, task_type)
    
    def optimize_ensemble_weights(self, models: List[BaseEstimator], 
                                 X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Optimize ensemble weights based on individual model performance"""
        try:
            logger.info("🔗 Optimizing ensemble weights...")
            
            weights = {}
            
            # Evaluate each model
            for i, model in enumerate(models):
                from sklearn.model_selection import cross_val_score
                
                if len(np.unique(y)) < 20:  # Classification
                    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                else:  # Regression
                    scores = cross_val_score(model, X, y, cv=3, scoring='r2')
                
                weights[f"model_{i}"] = scores.mean()
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            
            logger.info(f"✅ Ensemble weights optimized for {len(models)} models")
            return weights
            
        except Exception as e:
            logger.error(f"❌ Ensemble weight optimization failed: {e}")
            # Return equal weights as fallback
            return {f"model_{i}": 1.0/len(models) for i in range(len(models))}
    
    def evaluate_ensemble(self, ensemble: BaseEstimator, 
                         X_test: pd.DataFrame, y_test: pd.Series,
                         individual_models: List[BaseEstimator] = None) -> Dict[str, Any]:
        """Evaluate ensemble performance"""
        try:
            logger.info("🔗 Evaluating ensemble performance...")
            
            # Evaluate ensemble
            y_pred = ensemble.predict(X_test)
            
            if len(np.unique(y_test)) < 20:  # Classification
                from sklearn.metrics import accuracy_score, f1_score
                ensemble_score = accuracy_score(y_test, y_pred)
                ensemble_f1 = f1_score(y_test, y_pred, average='weighted')
                metrics = {'accuracy': ensemble_score, 'f1_score': ensemble_f1}
            else:  # Regression
                from sklearn.metrics import r2_score, mean_squared_error
                ensemble_score = r2_score(y_test, y_pred)
                ensemble_mse = mean_squared_error(y_test, y_pred)
                metrics = {'r2_score': ensemble_score, 'mse': ensemble_mse}
            
            results = {
                'ensemble_score': ensemble_score,
                'ensemble_metrics': metrics,
                'individual_scores': {}
            }
            
            # Evaluate individual models if provided
            if individual_models:
                for i, model in enumerate(individual_models):
                    y_pred_ind = model.predict(X_test)
                    
                    if len(np.unique(y_test)) < 20:  # Classification
                        ind_score = accuracy_score(y_test, y_pred_ind)
                    else:  # Regression
                        ind_score = r2_score(y_test, y_pred_ind)
                    
                    results['individual_scores'][f"model_{i}"] = ind_score
            
            logger.info(f"✅ Ensemble evaluation complete: {ensemble_score:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Ensemble evaluation failed: {e}")
            return {'error': str(e)}
    
    def _get_final_estimator(self, task_type: str) -> BaseEstimator:
        """Get final estimator for stacking"""
        try:
            if task_type == 'classification':
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(random_state=42, max_iter=1000)
            else:
                from sklearn.linear_model import LinearRegression
                return LinearRegression()
        except ImportError:
            # Fallback
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if task_type == 'classification':
                return RandomForestClassifier(n_estimators=10, random_state=42)
            else:
                return RandomForestRegressor(n_estimators=10, random_state=42)
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get summary of ensemble activities"""
        return {
            'available_methods': self.available_methods,
            'total_ensembles': len(self.ensemble_history),
            'latest_ensemble': self.ensemble_history[-1] if self.ensemble_history else None
        }
    
    def generate_ensemble_report(self) -> str:
        """Generate ensemble report"""
        lines = []
        
        # Header
        lines.append("🔗 AutoForge Ensemble Report")
        lines.append("=" * 50)
        
        # Method availability
        lines.append("\n📋 Available Methods:")
        for method, available in self.available_methods.items():
            status = "✅" if available else "❌"
            lines.append(f"  {status} {method}")
        
        # Ensemble history
        if self.ensemble_history:
            lines.append(f"\n🎯 Ensemble History:")
            lines.append(f"  Total Ensembles: {len(self.ensemble_history)}")
            
            latest = self.ensemble_history[-1]
            lines.append(f"  Latest Score: {latest.get('ensemble_score', 'N/A'):.4f}")
            lines.append(f"  Latest Method: {latest.get('method', 'N/A')}")
        
        return "\n".join(lines)


# Global ensemble integrator instance
ensemble_integrator = EnsembleIntegrator()


def create_ensemble_pipeline(models: List[BaseEstimator], X: pd.DataFrame, y: pd.Series,
                             method: str = 'stacking') -> BaseEstimator:
    """Create ensemble pipeline using available methods"""
    return ensemble_integrator.create_custom_ensemble(models, X, y, method)


def evaluate_ensemble_performance(ensemble: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series,
                                 individual_models: List[BaseEstimator] = None) -> Dict[str, Any]:
    """Evaluate ensemble performance"""
    return ensemble_integrator.evaluate_ensemble(ensemble, X_test, y_test, individual_models)
