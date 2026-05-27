"""
📊 Tracking Integration
Connects AutoForge with existing tracking modules
"""

import logging
import time
import json
import uuid
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

try:
    from .experiment import Experiment
    from .logger import ExperimentLogger
    from .storage import ExperimentStorage
except ImportError:
    Experiment = None
    ExperimentLogger = None
    ExperimentStorage = None

logger = logging.getLogger(__name__)


class TrackingIntegrator:
    """
    Integration layer for experiment tracking
    """
    
    def __init__(self):
        """Initialize tracking integrator"""
        self.available_components = self._check_available_components()
        self.experiments = []
        self.current_experiment = None
        
    def _check_available_components(self) -> Dict[str, bool]:
        """Check which tracking components are available"""
        components = {
            'experiment': Experiment is not None,
            'logger': ExperimentLogger is not None,
            'storage': ExperimentStorage is not None
        }
        
        available_count = sum(components.values())
        logger.info(f"📊 Available tracking components: {available_count}/{len(components)}")
        
        return components
    
    def start_experiment(self, autoforge_instance, experiment_name: str, 
                         description: str = "", tags: List[str] = None) -> str:
        """Start tracking an AutoForge experiment"""
        try:
            experiment_id = str(uuid.uuid4())
            
            logger.info(f"📊 Starting experiment: {experiment_name} (ID: {experiment_id})")
            
            # Create experiment metadata
            experiment_data = {
                'experiment_id': experiment_id,
                'name': experiment_name,
                'description': description,
                'tags': tags or [],
                'start_time': time.time(),
                'status': 'running',
                'autoforge_config': autoforge_instance.config,
                'system_info': self._get_system_info()
            }
            
            # Store experiment
            self.current_experiment = experiment_data
            self.experiments.append(experiment_data)
            
            # Log experiment start
            if self.available_components.get('logger', False):
                self._log_experiment_start(experiment_data)
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start experiment: {e}")
            return None
    
    def log_training_step(self, step_name: str, metrics: Dict[str, Any], 
                         parameters: Dict[str, Any] = None):
        """Log a training step"""
        try:
            if not self.current_experiment:
                logger.warning("⚠️ No active experiment to log to")
                return
            
            step_data = {
                'step_name': step_name,
                'timestamp': time.time(),
                'metrics': metrics,
                'parameters': parameters or {}
            }
            
            # Add to current experiment
            if 'steps' not in self.current_experiment:
                self.current_experiment['steps'] = []
            self.current_experiment['steps'].append(step_data)
            
            # Log to external logger if available
            if self.available_components.get('logger', False):
                self._log_training_step(step_data)
            
            logger.debug(f"📊 Logged step: {step_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log training step: {e}")
    
    def log_autoforge_training(self, autoforge_instance, X: pd.DataFrame, y: pd.Series):
        """Log complete AutoForge training process"""
        try:
            if not self.current_experiment:
                logger.warning("⚠️ No active experiment to log to")
                return
            
            logger.info("📊 Logging AutoForge training...")
            
            # Log dataset info
            self.log_training_step('dataset_info', {
                'dataset_shape': X.shape,
                'target_type': 'classification' if len(np.unique(y)) < 20 else 'regression',
                'target_classes': len(np.unique(y)),
                'missing_values': X.isnull().sum().sum(),
                'feature_types': {
                    'numeric': len(X.select_dtypes(include=[np.number]).columns),
                    'categorical': len(X.select_dtypes(include=['object', 'category']).columns)
                }
            })
            
            # Log intelligence analysis
            if hasattr(autoforge_instance, 'current_profile') and autoforge_instance.current_profile:
                profile = autoforge_instance.current_profile
                self.log_training_step('intelligence_analysis', {
                    'data_type': profile.data_type,
                    'task_type': profile.complexity_profile.get('task_type', 'unknown'),
                    'strategy_used': profile.recommendations.get('primary_strategy', 'unknown'),
                    'confidence': profile.confidence
                })
            
            # Log feature selection
            if hasattr(autoforge_instance, 'feature_selector'):
                self.log_training_step('feature_selection', {
                    'original_features': X.shape[1],
                    'selected_features': len(autoforge_instance.training_metadata.get('selected_features', [])),
                    'selection_ratio': autoforge_instance.training_metadata.get('feature_selection_ratio', 0)
                })
            
            # Log model training
            self.log_training_step('model_training', {
                'best_score': autoforge_instance.best_score,
                'model_type': type(autoforge_instance.best_model).__name__,
                'training_time': autoforge_instance.training_metadata.get('training_time', 0),
                'models_tried': list(autoforge_instance.training_metadata.get('training_results', {}).keys())
            })
            
            # Log evaluation
            if hasattr(autoforge_instance, 'evaluator'):
                # Simple evaluation
                predictions = autoforge_instance.predict(X)
                if len(np.unique(y)) < 20:  # Classification
                    from sklearn.metrics import accuracy_score, f1_score
                    accuracy = accuracy_score(y, predictions)
                    f1 = f1_score(y, predictions, average='weighted')
                    eval_metrics = {'accuracy': accuracy, 'f1_score': f1}
                else:  # Regression
                    from sklearn.metrics import r2_score, mean_squared_error
                    r2 = r2_score(y, predictions)
                    mse = mean_squared_error(y, predictions)
                    eval_metrics = {'r2_score': r2, 'mse': mse}
                
                self.log_training_step('evaluation', eval_metrics)
            
            logger.info("📊 AutoForge training logged successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to log AutoForge training: {e}")
    
    def end_experiment(self, status: str = 'completed', final_metrics: Dict[str, Any] = None):
        """End the current experiment"""
        try:
            if not self.current_experiment:
                logger.warning("⚠️ No active experiment to end")
                return
            
            logger.info(f"📊 Ending experiment: {self.current_experiment['name']}")
            
            # Update experiment
            self.current_experiment['end_time'] = time.time()
            self.current_experiment['duration'] = self.current_experiment['end_time'] - self.current_experiment['start_time']
            self.current_experiment['status'] = status
            self.current_experiment['final_metrics'] = final_metrics or {}
            
            # Log experiment end
            if self.available_components.get('logger', False):
                self._log_experiment_end(self.current_experiment)
            
            # Save experiment if storage available
            if self.available_components.get('storage', False):
                self._save_experiment(self.current_experiment)
            
            logger.info(f"📊 Experiment ended: {status}")
            
        except Exception as e:
            logger.error(f"❌ Failed to end experiment: {e}")
    
    def get_experiment_summary(self, experiment_id: str = None) -> Dict[str, Any]:
        """Get summary of an experiment"""
        try:
            if experiment_id:
                experiment = next((exp for exp in self.experiments if exp['experiment_id'] == experiment_id), None)
            else:
                experiment = self.current_experiment
            
            if not experiment:
                return {'error': 'Experiment not found'}
            
            summary = {
                'experiment_id': experiment['experiment_id'],
                'name': experiment['name'],
                'status': experiment['status'],
                'start_time': experiment['start_time'],
                'duration': experiment.get('duration', 0),
                'steps_count': len(experiment.get('steps', [])),
                'final_metrics': experiment.get('final_metrics', {})
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get experiment summary: {e}")
            return {'error': str(e)}
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments"""
        return self.experiments
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            import platform
            import psutil
            
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'disk_usage': psutil.disk_usage('/').percent
            }
        except ImportError:
            return {'platform': 'Unknown', 'python_version': 'Unknown'}
    
    def _log_experiment_start(self, experiment_data: Dict[str, Any]):
        """Log experiment start to external logger"""
        try:
            if ExperimentLogger:
                logger = ExperimentLogger()
                logger.log_experiment_start(experiment_data)
        except Exception as e:
            logger.warning(f"External logging failed: {e}")
    
    def _log_training_step(self, step_data: Dict[str, Any]):
        """Log training step to external logger"""
        try:
            if ExperimentLogger:
                logger = ExperimentLogger()
                logger.log_step(step_data)
        except Exception as e:
            logger.warning(f"External step logging failed: {e}")
    
    def _log_experiment_end(self, experiment_data: Dict[str, Any]):
        """Log experiment end to external logger"""
        try:
            if ExperimentLogger:
                logger = ExperimentLogger()
                logger.log_experiment_end(experiment_data)
        except Exception as e:
            logger.warning(f"External end logging failed: {e}")
    
    def _save_experiment(self, experiment_data: Dict[str, Any]):
        """Save experiment to external storage"""
        try:
            if ExperimentStorage:
                storage = ExperimentStorage()
                storage.save_experiment(experiment_data)
        except Exception as e:
            logger.warning(f"External storage failed: {e}")
    
    def generate_tracking_report(self) -> str:
        """Generate tracking report"""
        lines = []
        
        # Header
        lines.append("📊 AutoForge Tracking Report")
        lines.append("=" * 50)
        
        # Component availability
        lines.append("\n📋 Available Components:")
        for component, available in self.available_components.items():
            status = "✅" if available else "❌"
            lines.append(f"  {status} {component}")
        
        # Experiment summary
        lines.append(f"\n🧪 Experiment Summary:")
        lines.append(f"  Total Experiments: {len(self.experiments)}")
        
        if self.experiments:
            completed = sum(1 for exp in self.experiments if exp.get('status') == 'completed')
            running = sum(1 for exp in self.experiments if exp.get('status') == 'running')
            failed = sum(1 for exp in self.experiments if exp.get('status') == 'failed')
            
            lines.append(f"  Completed: {completed}")
            lines.append(f"  Running: {running}")
            lines.append(f"  Failed: {failed}")
        
        # Current experiment
        if self.current_experiment:
            lines.append(f"\n🎯 Current Experiment:")
            lines.append(f"  Name: {self.current_experiment['name']}")
            lines.append(f"  Status: {self.current_experiment['status']}")
            lines.append(f"  Steps: {len(self.current_experiment.get('steps', []))}")
            
            if self.current_experiment.get('final_metrics'):
                lines.append(f"  Final Score: {self.current_experiment['final_metrics'].get('best_score', 'N/A')}")
        
        return "\n".join(lines)


# Global tracking integrator instance
tracking_integrator = TrackingIntegrator()


def start_autoforge_experiment(autoforge_instance, experiment_name: str, 
                            description: str = "", tags: List[str] = None) -> str:
    """Start tracking an AutoForge experiment"""
    return tracking_integrator.start_experiment(autoforge_instance, experiment_name, description, tags)


def log_autoforge_training(autoforge_instance, X: pd.DataFrame, y: pd.Series):
    """Log complete AutoForge training process"""
    tracking_integrator.log_autoforge_training(autoforge_instance, X, y)


def end_autoforge_experiment(status: str = 'completed', final_metrics: Dict[str, Any] = None):
    """End the current AutoForge experiment"""
    tracking_integrator.end_experiment(status, final_metrics)
