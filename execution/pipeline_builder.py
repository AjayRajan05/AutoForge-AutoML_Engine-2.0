"""
🔧 Pipeline Builder - Construct ML pipelines from components
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# Import existing pipeline builder if available
try:
    # Try to import from registry if available
    from ..registry.pipeline_registry import build_pipeline as original_build_pipeline
except ImportError:
    original_build_pipeline = None

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """
    Build ML pipelines from preprocessing steps and models
    """
    
    def __init__(self):
        """Initialize pipeline builder"""
        self.preprocessors = {}
        self.transformers = {}
        self.pipelines = {}
        
    def build_pipeline(self, X: pd.DataFrame, y: pd.Series, 
                       preprocessing_steps: List[str],
                       model: Any,
                       config: Dict[str, Any]) -> Pipeline:
        """
        Build a complete ML pipeline
        
        Args:
            X: Feature data
            y: Target data
            preprocessing_steps: List of preprocessing steps
            model: Model to use
            config: Configuration
            
        Returns:
            Scikit-learn Pipeline
        """
        try:
            logger.info("🔧 Building ML pipeline...")
            
            # Try to use existing pipeline builder first
            if original_build_pipeline:
                try:
                    # Extract parameters for existing builder
                    params = self._extract_pipeline_params(config, X)
                    
                    # Use existing pipeline builder
                    pipeline = original_build_pipeline(params, model)
                    
                    logger.info("✅ Used existing pipeline builder")
                    return pipeline
                    
                except Exception as e:
                    logger.warning(f"⚠️ Existing pipeline builder failed: {e}")
            
            # Fallback to our enhanced pipeline builder
            return self._build_enhanced_pipeline(X, y, preprocessing_steps, model, config)
            
        except Exception as e:
            logger.error(f"❌ Pipeline building failed: {e}")
            # Return simple pipeline as fallback
            return Pipeline([('model', model)])
    
    def _extract_pipeline_params(self, config: Dict[str, Any], X: pd.DataFrame) -> Dict[str, Any]:
        """Extract parameters for existing pipeline builder"""
        # Use configurable pipeline builder defaults
        try:
            from config.settings import get_config_value
            default_imputation = get_config_value('pipeline', 'default_imputation', 'mean')
            default_scaler = get_config_value('pipeline', 'default_scaler', 'standard')
            default_max_features = get_config_value('pipeline', 'default_max_features', X.shape[1])
        except ImportError:
            default_imputation = 'mean'
            default_scaler = 'standard'
            default_max_features = X.shape[1]
        
        params = {}
        
        # Imputation
        if config.get('handle_missing', True):
            params['imputer'] = config.get('imputation_strategy', default_imputation)
        
        # Scaling
        if config.get('scale_features', True):
            params['scaler'] = config.get('scaler_type', default_scaler)
        
        # Feature selection
        if config.get('feature_selection', False):
            params['feature_selection'] = True
            params['k_best'] = config.get('max_features', default_max_features)
        
        return params
    
    def _build_enhanced_pipeline(self, X: pd.DataFrame, y: pd.Series, 
                                 preprocessing_steps: List[str],
                                 model: Any,
                                 config: Dict[str, Any]) -> Pipeline:
        """Build enhanced pipeline with AutoForge features"""
        try:
            # Identify column types
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Build preprocessing steps
            preprocessing_pipeline = self._build_preprocessing_pipeline(
                numeric_features, categorical_features, preprocessing_steps, config
            )
            
            # Build complete pipeline
            if preprocessing_pipeline:
                pipeline_steps = [('preprocessing', preprocessing_pipeline)]
            else:
                pipeline_steps = []
            
            pipeline_steps.append(('model', model))
            
            pipeline = Pipeline(pipeline_steps)
            
            logger.info(f"✅ Enhanced pipeline built with {len(pipeline_steps)} steps")
            return pipeline
            
        except Exception as e:
            logger.error(f"❌ Enhanced pipeline building failed: {e}")
            # Return simple pipeline as fallback
            return Pipeline([('model', model)])
    
    def _build_preprocessing_pipeline(self, numeric_features: List[str], 
                                    categorical_features: List[str],
                                    steps: List[str], 
                                    config: Dict[str, Any]) -> Optional[ColumnTransformer]:
        """Build preprocessing pipeline"""
        transformers = []
        
        # Numeric preprocessing
        if numeric_features and any(step in steps for step in ['scaling', 'imputation', 'normalization']):
            numeric_transformer = self._build_numeric_transformer(numeric_features, steps, config)
            if numeric_transformer:
                transformers.append(('numeric', numeric_transformer, numeric_features))
        
        # Categorical preprocessing
        if categorical_features and any(step in steps for step in ['encoding', 'imputation']):
            categorical_transformer = self._build_categorical_transformer(categorical_features, steps, config)
            if categorical_transformer:
                transformers.append(('categorical', categorical_transformer, categorical_features))
        
        if transformers:
            return ColumnTransformer(transformers=transformers, remainder='passthrough')
        else:
            return None
    
    def _build_numeric_transformer(self, features: List[str], steps: List[str], 
                                 config: Dict[str, Any]) -> Pipeline:
        """Build numeric feature transformer"""
        # Use configurable numeric transformer defaults
        try:
            from config.settings import get_config_value
            default_numeric_imputation = get_config_value('pipeline', 'default_numeric_imputation', 'mean')
        except ImportError:
            default_numeric_imputation = 'mean'
        
        transformer_steps = []
        
        # Imputation
        if 'imputation' in steps:
            strategy = config.get('imputation_strategy', default_numeric_imputation)
            transformer_steps.append(('imputer', SimpleImputer(strategy=strategy)))
        
        # Scaling
        if 'scaling' in steps or 'normalization' in steps:
            transformer_steps.append(('scaler', StandardScaler()))
        
        return Pipeline(transformer_steps) if transformer_steps else None
    
    def _build_categorical_transformer(self, features: List[str], steps: List[str], 
                                     config: Dict[str, Any]) -> Pipeline:
        """Build categorical feature transformer"""
        # Use configurable categorical transformer defaults
        try:
            from config.settings import get_config_value
            default_categorical_imputation = get_config_value('pipeline', 'default_categorical_imputation', 'most_frequent')
            default_encoding_type = get_config_value('pipeline', 'default_encoding_type', 'onehot')
        except ImportError:
            default_categorical_imputation = 'most_frequent'
            default_encoding_type = 'onehot'
        
        transformer_steps = []
        
        # Imputation
        if 'imputation' in steps:
            strategy = config.get('categorical_imputation_strategy', default_categorical_imputation)
            transformer_steps.append(('imputer', SimpleImputer(strategy=strategy)))
        
        # Encoding
        if 'encoding' in steps:
            encoding_type = config.get('encoding_type', default_encoding_type)
            if encoding_type == 'onehot':
                transformer_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore')))
            else:
                transformer_steps.append(('encoder', LabelEncoder()))
        
        return Pipeline(transformer_steps) if transformer_steps else None
    
    def get_pipeline_summary(self, pipeline: Pipeline) -> Dict[str, Any]:
        """Get summary of pipeline components"""
        summary = {
            'steps': [],
            'total_steps': len(pipeline.steps),
            'has_preprocessing': False,
            'preprocessing_type': None
        }
        
        for name, step in pipeline.steps:
            step_info = {
                'name': name,
                'type': type(step).__name__,
                'parameters': step.get_params() if hasattr(step, 'get_params') else {}
            }
            summary['steps'].append(step_info)
            
            if name == 'preprocessing':
                summary['has_preprocessing'] = True
                summary['preprocessing_type'] = type(step).__name__
        
        return summary
    
    def optimize_pipeline(self, X: pd.DataFrame, y: pd.Series,
                         base_pipeline: Pipeline,
                         config: Dict[str, Any]) -> Pipeline:
        """Optimize pipeline based on data characteristics"""
        try:
            logger.info("⚡ Optimizing pipeline...")
            
            # Analyze data characteristics
            missing_ratio = X.isnull().sum().sum() / X.size
            categorical_ratio = len(X.select_dtypes(include=['object', 'category']).columns) / X.shape[1]
            
            # Adjust pipeline based on characteristics
            optimized_steps = []
            
            for name, step in base_pipeline.steps:
                if name == 'preprocessing' and hasattr(step, 'transformers'):
                    # Optimize preprocessing
                    optimized_transformers = []
                    
                    for transformer_name, transformer, columns in step.transformers:
                        # Adjust based on data characteristics
                        if missing_ratio < 0.05 and 'imputer' in str(type(transformer)):
                            # Skip imputation if very few missing values
                            continue
                        
                        if categorical_ratio < 0.1 and transformer_name == 'categorical':
                            # Simplify categorical processing for few categorical features
                            continue
                        
                        optimized_transformers.append((transformer_name, transformer, columns))
                    
                    if optimized_transformers:
                        from sklearn.compose import ColumnTransformer
                        optimized_step = ColumnTransformer(
                            transformers=optimized_transformers, 
                            remainder='passthrough'
                        )
                        optimized_steps.append((name, optimized_step))
                    else:
                        # Skip preprocessing entirely
                        continue
                else:
                    optimized_steps.append((name, step))
            
            optimized_pipeline = Pipeline(optimized_steps)
            
            logger.info(f"✅ Pipeline optimized: {len(optimized_steps)} steps (was {len(base_pipeline.steps)})")
            return optimized_pipeline
            
        except Exception as e:
            logger.warning(f"⚠️ Pipeline optimization failed: {e}")
            return base_pipeline
