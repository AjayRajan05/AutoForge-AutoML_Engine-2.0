"""
💾 Model Persistence - Save and load trained models with joblib
"""

import logging
import joblib
import os
import json
import pickle
from typing import Any, Dict, Optional, Union
from pathlib import Path
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelSaver:
    """
    Model persistence system for AutoForge
    
    Handles saving and loading of trained models, metadata, and related artifacts
    """
    
    def __init__(self, base_path: str = "./models"):
        """
        Initialize model saver
        
        Args:
            base_path: Base directory for saving models
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Supported formats
        self.supported_formats = {
            'joblib': self._save_joblib,
            'pickle': self._save_pickle,
            'json': self._save_json_metadata
        }
        
        self.supported_loaders = {
            'joblib': self._load_joblib,
            'pickle': self._load_pickle,
            'json': self._load_json_metadata
        }
        
        logger.info(f"💾 ModelSaver initialized with base path: {self.base_path}")
    
    def save_model(self, model: Any, 
                   model_name: str,
                   metadata: Optional[Dict[str, Any]] = None,
                   format: str = 'joblib',
                   overwrite: bool = False) -> str:
        """
        Save a trained model with metadata
        
        Args:
            model: Trained model object
            model_name: Name for the saved model
            metadata: Additional metadata to save
            format: Save format ('joblib', 'pickle')
            overwrite: Whether to overwrite existing model
            
        Returns:
            Path to saved model
        """
        try:
            logger.info(f"💾 Saving model: {model_name}")
            
            # Create model directory
            model_dir = self.base_path / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Check if model exists
            model_path = model_dir / f"model.{format}"
            if model_path.exists() and not overwrite:
                raise FileExistsError(f"Model {model_name} already exists. Use overwrite=True to replace.")
            
            # Save model
            if format not in self.supported_formats:
                raise ValueError(f"Unsupported format: {format}. Supported: {list(self.supported_formats.keys())}")
            
            self.supported_formats[format](model, model_path)
            
            # Prepare metadata
            full_metadata = {
                'model_name': model_name,
                'save_timestamp': datetime.now().isoformat(),
                'format': format,
                'model_type': type(model).__name__,
                'model_module': type(model).__module__,
                'file_path': str(model_path),
                'file_size_bytes': model_path.stat().st_size if model_path.exists() else 0
            }
            
            # Add user metadata
            if metadata:
                full_metadata.update(metadata)
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2, default=str)
            
            logger.info(f"✅ Model saved successfully: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save model {model_name}: {e}")
            raise
    
    def load_model(self, model_name: str, 
                   format: Optional[str] = None) -> tuple[Any, Dict[str, Any]]:
        """
        Load a saved model with metadata
        
        Args:
            model_name: Name of the saved model
            format: Load format (auto-detected if None)
            
        Returns:
            Tuple of (model, metadata)
        """
        try:
            logger.info(f"📂 Loading model: {model_name}")
            
            model_dir = self.base_path / model_name
            
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
            # Load metadata first
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Auto-detect format if not specified
            if format is None:
                format = metadata.get('format', 'joblib')
            
            # Find model file
            model_path = model_dir / f"model.{format}"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model
            if format not in self.supported_loaders:
                raise ValueError(f"Unsupported format: {format}")
            
            model = self.supported_loaders[format](model_path)
            
            logger.info(f"✅ Model loaded successfully: {model_path}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            raise
    
    def save_pipeline(self, model: Any,
                     preprocessing_steps: Optional[Dict[str, Any]] = None,
                     feature_info: Optional[Dict[str, Any]] = None,
                     training_info: Optional[Dict[str, Any]] = None,
                     pipeline_name: str = "autoforge_pipeline",
                     report_bundle: Optional[Union[bool, Dict[str, str]]] = None) -> str:
        """
        Save complete pipeline with model, preprocessing, and metadata
        
        Args:
            model: Trained model
            preprocessing_steps: Preprocessing pipeline information
            feature_info: Feature engineering information
            training_info: Training metrics and information
            pipeline_name: Name for the pipeline
            
        Returns:
            Path to saved pipeline
        """
        try:
            logger.info(f"🔧 Saving complete pipeline: {pipeline_name}")
            
            # Create pipeline directory
            pipeline_dir = self.base_path / pipeline_name
            pipeline_dir.mkdir(exist_ok=True)
            
            # Save model
            backend = getattr(model, 'backend', 'sklearn')
            if backend == 'keras' and hasattr(model, 'model'):
                model_dir = self.base_path / f"{pipeline_name}/model"
                model_dir.mkdir(parents=True, exist_ok=True)
                keras_path = model_dir / "model.h5"
                model.model.save(str(keras_path))
                model_path = str(keras_path)
                pipeline_metadata = {
                    'pipeline_name': pipeline_name,
                    'save_timestamp': datetime.now().isoformat(),
                    'components': {
                        'model': 'model/model.h5',
                        'preprocessing': 'preprocessing.json',
                        'features': 'features.json',
                        'training': 'training.json'
                    },
                    'autoforge_version': '1.0.0',
                    'model_backend': 'keras',
                    'task_type': (training_info or {}).get('task_type', 'classification'),
                }
            else:
                model_path = self.save_model(
                    model,
                    f"{pipeline_name}/model",
                    metadata={'component': 'model'},
                    overwrite=True
                )
                pipeline_metadata = {
                    'pipeline_name': pipeline_name,
                    'save_timestamp': datetime.now().isoformat(),
                    'components': {
                        'model': 'model/model.joblib',
                        'preprocessing': 'preprocessing.json',
                        'features': 'features.json',
                        'training': 'training.json'
                    },
                    'autoforge_version': '1.0.0',
                    'model_backend': 'sklearn',
                }
            
            # Save preprocessing steps (JSON metadata + joblib for sklearn objects)
            if preprocessing_steps:
                serializable = {}
                artifacts = {}
                sklearn_keys = {
                    'scaler', 'target_encoder', 'categorical_encoding', 'column_transformer',
                }
                for key, value in preprocessing_steps.items():
                    if key in sklearn_keys and value is not None:
                        artifacts[key] = value
                    else:
                        serializable[key] = value

                prep_path = pipeline_dir / "preprocessing.json"
                with open(prep_path, 'w') as f:
                    json.dump(serializable, f, indent=2, default=str)

                if artifacts:
                    joblib.dump(artifacts, pipeline_dir / "preprocessing_artifacts.joblib")
            
            # Save feature information
            if feature_info:
                feat_path = pipeline_dir / "features.json"
                with open(feat_path, 'w') as f:
                    json.dump(feature_info, f, indent=2, default=str)
            
            # Save training information
            if training_info:
                train_path = pipeline_dir / "training.json"
                with open(train_path, 'w') as f:
                    json.dump(training_info, f, indent=2, default=str)
            
            # Optional justification report files (paths supplied by caller)
            if isinstance(report_bundle, dict):
                for filename, content in report_bundle.items():
                    dest = pipeline_dir / filename
                    if isinstance(content, (dict, list)):
                        with open(dest, 'w', encoding='utf-8') as f:
                            json.dump(content, f, indent=2, default=str)
                    else:
                        dest.write_text(str(content), encoding='utf-8')

            # Save pipeline metadata
            metadata_path = pipeline_dir / "pipeline.json"
            with open(metadata_path, 'w') as f:
                json.dump(pipeline_metadata, f, indent=2, default=str)
            
            logger.info(f"✅ Pipeline saved successfully: {pipeline_dir}")
            return str(pipeline_dir)
            
        except Exception as e:
            logger.error(f"❌ Failed to save pipeline {pipeline_name}: {e}")
            raise
    
    def load_pipeline(self, pipeline_name: str) -> Dict[str, Any]:
        """
        Load complete pipeline
        
        Args:
            pipeline_name: Name of the saved pipeline
            
        Returns:
            Dictionary with model and all components
        """
        try:
            logger.info(f"📂 Loading complete pipeline: {pipeline_name}")
            
            pipeline_dir = self.base_path / pipeline_name
            
            if not pipeline_dir.exists():
                raise FileNotFoundError(f"Pipeline directory not found: {pipeline_dir}")
            
            # Load pipeline metadata
            metadata_path = pipeline_dir / "pipeline.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Pipeline metadata not found: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                pipeline_metadata = json.load(f)
            
            # Load model (sklearn joblib or Keras h5)
            if pipeline_metadata.get('model_backend') == 'keras':
                try:
                    from tensorflow import keras
                    from execution.dl_trainer import KerasModelWrapper
                    keras_path = pipeline_dir / "model" / "model.h5"
                    keras_model = keras.models.load_model(str(keras_path))
                    task_type = pipeline_metadata.get('task_type', 'classification')
                    model = KerasModelWrapper(keras_model, task_type=task_type)
                except ImportError as exc:
                    raise ImportError(
                        "Loading Keras models requires: pip install 'autoforge[dl]'"
                    ) from exc
            else:
                model, model_metadata = self.load_model(f"{pipeline_name}/model")
            
            # Load other components
            components = {'model': model}
            
            # Load preprocessing
            prep_path = pipeline_dir / "preprocessing.json"
            if prep_path.exists():
                with open(prep_path, 'r') as f:
                    components['preprocessing'] = json.load(f)

            artifacts_path = pipeline_dir / "preprocessing_artifacts.joblib"
            if artifacts_path.exists():
                artifacts = joblib.load(artifacts_path)
                components.setdefault('preprocessing', {}).update(artifacts)
            
            # Load features
            feat_path = pipeline_dir / "features.json"
            if feat_path.exists():
                with open(feat_path, 'r') as f:
                    components['features'] = json.load(f)
            
            # Load training info
            train_path = pipeline_dir / "training.json"
            if train_path.exists():
                with open(train_path, 'r') as f:
                    components['training'] = json.load(f)
            
            components['metadata'] = pipeline_metadata
            
            logger.info(f"✅ Pipeline loaded successfully: {pipeline_dir}")
            return components
            
        except Exception as e:
            logger.error(f"❌ Failed to load pipeline {pipeline_name}: {e}")
            raise
    
    def list_saved_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all saved models with their metadata
        
        Returns:
            Dictionary of model_name -> metadata
        """
        try:
            models = {}
            
            for model_dir in self.base_path.iterdir():
                if model_dir.is_dir():
                    metadata_path = model_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        models[model_dir.name] = metadata
            
            logger.info(f"📋 Found {len(models)} saved models")
            return models
            
        except Exception as e:
            logger.error(f"❌ Failed to list models: {e}")
            return {}
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a saved model
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            model_dir = self.base_path / model_name
            
            if not model_dir.exists():
                logger.warning(f"Model not found: {model_name}")
                return False
            
            # Remove directory and all contents
            import shutil
            shutil.rmtree(model_dir)
            
            logger.info(f"🗑️  Model deleted successfully: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete model {model_name}: {e}")
            return False
    
    def _save_joblib(self, model: Any, path: Path):
        """Save model using joblib"""
        joblib.dump(model, path, compress=3)
    
    def _save_pickle(self, model: Any, path: Path):
        """Save model using pickle"""
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    def _save_json_metadata(self, metadata: Dict[str, Any], path: Path):
        """Save metadata as JSON"""
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _load_joblib(self, path: Path) -> Any:
        """Load model using joblib"""
        return joblib.load(path)
    
    def _load_pickle(self, path: Path) -> Any:
        """Load model using pickle"""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _load_json_metadata(self, path: Path) -> Dict[str, Any]:
        """Load metadata from JSON"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a saved model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model metadata or None if not found
        """
        try:
            model_dir = self.base_path / model_name
            
            if not model_dir.exists():
                return None
            
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Add additional info
            metadata['directory_exists'] = True
            metadata['model_file_exists'] = (model_dir / f"model.{metadata.get('format', 'joblib')}").exists()
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to get model info for {model_name}: {e}")
            return None
    
    def cleanup_old_models(self, max_models: int = 10) -> int:
        """
        Clean up old models, keeping only the most recent ones
        
        Args:
            max_models: Maximum number of models to keep
            
        Returns:
            Number of models deleted
        """
        try:
            models = self.list_saved_models()
            
            if len(models) <= max_models:
                logger.info(f"📋 No cleanup needed: {len(models)} models (max: {max_models})")
                return 0
            
            # Sort by save timestamp
            sorted_models = sorted(
                models.items(),
                key=lambda x: x[1].get('save_timestamp', ''),
                reverse=True
            )
            
            # Delete oldest models
            models_to_delete = sorted_models[max_models:]
            deleted_count = 0
            
            for model_name, metadata in models_to_delete:
                if self.delete_model(model_name):
                    deleted_count += 1
            
            logger.info(f"🗑️  Cleaned up {deleted_count} old models")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old models: {e}")
            return 0


# Global instance for convenience
model_saver = ModelSaver()
