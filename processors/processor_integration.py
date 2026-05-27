"""
🔧 Processor Integration
Connects AutoForge with existing processor modules
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union

try:
    from .tabular_processor import TabularProcessor
except ImportError:
    TabularProcessor = None

try:
    from execution.preprocessing_pipeline import PreprocessingPipeline
except ImportError:
    PreprocessingPipeline = None

try:
    from .text_processor import TextProcessor
except ImportError:
    TextProcessor = None

try:
    from .image_processor import ImageProcessor
except ImportError:
    ImageProcessor = None

try:
    from .time_series_processor import TimeSeriesProcessor
except ImportError:
    TimeSeriesProcessor = None

logger = logging.getLogger(__name__)


class ProcessorIntegrator:
    """
    Integration layer for data processors
    """
    
    def __init__(self):
        """Initialize processor integrator"""
        self.available_processors = self._check_available_processors()
        self.processing_history = []
        
    def _check_available_processors(self) -> Dict[str, bool]:
        """Check which processors are available"""
        processors = {
            'tabular': TabularProcessor is not None,
            'text': TextProcessor is not None,
            'image': ImageProcessor is not None,
            'time_series': TimeSeriesProcessor is not None
        }
        
        available_count = sum(processors.values())
        logger.info(f"🔧 Available processors: {available_count}/{len(processors)}")
        
        return processors
    
    def detect_data_type(self, X: pd.DataFrame, y: pd.Series = None) -> str:
        """Detect the type of data for appropriate processor selection"""
        try:
            # Check for text data
            text_columns = X.select_dtypes(include=['object', 'category']).columns
            if len(text_columns) > 0:
                # Check if text columns contain actual text (not just categorical)
                sample_text = X[text_columns[0]].dropna().iloc[0] if len(text_columns) > 0 else ""
                if isinstance(sample_text, str) and len(sample_text.split()) > 3:
                    return 'text'
            
            # Check for time series data
            if y is not None and len(y) > 10:
                # Check for temporal patterns
                y_diff = np.diff(y.values)
                if np.std(y_diff) < np.std(y.values) * 0.1:  # Low variance in differences
                    return 'time_series'
            
            # Check for image data (simplified)
            if X.shape[1] > 100 and all(X.dtypes == np.number):
                return 'image'
            
            # Default to tabular
            return 'tabular'
            
        except Exception as e:
            logger.warning(f"⚠️ Data type detection failed: {e}")
            return 'tabular'
    
    def process_data(self, X: pd.DataFrame, y: pd.Series = None, 
                    data_type: str = None, config: Dict[str, Any] = None) -> tuple:
        """Process data using appropriate processor.

        Returns (X_processed, y_processed) or (X, y, artifacts) when return_artifacts=True.
        """
        try:
            if data_type is None:
                data_type = self.detect_data_type(X, y)
            
            logger.info(f"🔧 Processing {data_type} data...")
            
            config = config or {}
            return_artifacts = config.get('return_artifacts', False)
            
            if data_type == 'tabular' and PreprocessingPipeline is not None:
                result = self._process_tabular_pipeline(X, y, config)
            elif data_type == 'tabular' and self.available_processors.get('tabular', False):
                result = self._process_tabular(X, y, config)
            elif data_type == 'text' and self.available_processors.get('text', False):
                result = self._process_text(X, y, config)
            elif data_type == 'image' and self.available_processors.get('image', False):
                result = self._process_image(X, y, config)
            elif data_type == 'time_series' and self.available_processors.get('time_series', False):
                result = self._process_time_series(X, y, config)
            else:
                result = self._fallback_processing(X, y, data_type)

            if return_artifacts and isinstance(result, tuple) and len(result) == 3:
                return result
            if isinstance(result, tuple) and len(result) >= 2:
                return result[0], result[1]
            return result
                
        except Exception as e:
            logger.error(f"❌ Data processing failed: {e}")
            return X, y

    def _process_tabular_pipeline(self, X: pd.DataFrame, y: pd.Series,
                                  config: Dict[str, Any]) -> tuple:
        """Process tabular data via unified PreprocessingPipeline."""
        try:
            task_type = config.get('task_type') or 'classification'
            pipeline = PreprocessingPipeline(
                scale_features=config.get('scale_features', True),
                feature_engineering=config.get('feature_engineering', False),
            )
            X_processed, y_processed, artifacts = pipeline.fit_transform(
                X, y, task_type=task_type
            )
            self.processing_history.append({
                'data_type': 'tabular',
                'original_shape': X.shape,
                'processed_shape': X_processed.shape,
                'pipeline': 'PreprocessingPipeline',
                'timestamp': pd.Timestamp.now().isoformat(),
            })
            if config.get('return_artifacts'):
                return X_processed, y_processed, artifacts
            return X_processed, y_processed
        except Exception as exc:
            logger.warning("PreprocessingPipeline failed: %s", exc)
            return self._process_tabular(X, y, config)
    
    def _process_tabular(self, X: pd.DataFrame, y: pd.Series, config: Dict[str, Any]) -> tuple:
        """Process tabular data"""
        try:
            logger.info("🔧 Using TabularProcessor...")
            
            processor = TabularProcessor()
            
            # Setup processing configuration
            processing_config = {
                'handle_missing': config.get('handle_missing', True),
                'scale_features': config.get('scale_features', True),
                'encode_categorical': config.get('encode_categorical', True),
                'feature_engineering': config.get('feature_engineering', False),
                'remove_outliers': config.get('remove_outliers', False)
            }
            
            X_processed, _metadata = processor.process(X, y, processing_config)
            y_processed = y
            
            # Store processing history
            self.processing_history.append({
                'data_type': 'tabular',
                'original_shape': X.shape,
                'processed_shape': X_processed.shape,
                'config': processing_config,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            logger.info(f"✅ Tabular processing complete: {X.shape} → {X_processed.shape}")
            return X_processed, y_processed
            
        except Exception as e:
            logger.warning(f"⚠️ Tabular processing failed: {e}")
            return self._fallback_processing(X, y, 'tabular')
    
    def _process_text(self, X: pd.DataFrame, y: pd.Series, config: Dict[str, Any]) -> tuple:
        """Process text data"""
        try:
            logger.info("🔧 Using TextProcessor...")
            
            processor = TextProcessor()
            
            # Setup processing configuration
            processing_config = {
                'vectorization_method': config.get('vectorization', 'tfidf'),
                'max_features': config.get('max_features', 1000),
                'ngram_range': config.get('ngram_range', (1, 2)),
                'min_df': config.get('min_df', 2),
                'max_df': config.get('max_df', 0.95),
                'clean_text': config.get('clean_text', True),
                'remove_stopwords': config.get('remove_stopwords', True)
            }
            
            X_processed, _metadata = processor.process(X, y, processing_config)
            y_processed = y
            
            # Store processing history
            self.processing_history.append({
                'data_type': 'text',
                'original_shape': X.shape,
                'processed_shape': X_processed.shape,
                'config': processing_config,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            logger.info(f"✅ Text processing complete: {X.shape} → {X_processed.shape}")
            return X_processed, y_processed
            
        except Exception as e:
            logger.warning(f"⚠️ Text processing failed: {e}")
            return self._fallback_processing(X, y, 'text')
    
    def _process_image(self, X: pd.DataFrame, y: pd.Series, config: Dict[str, Any]) -> tuple:
        """Process image data"""
        try:
            logger.info("🔧 Using ImageProcessor...")
            
            processor = ImageProcessor()
            
            # Setup processing configuration
            processing_config = {
                'resize_images': config.get('resize_images', (64, 64)),
                'normalize_pixels': config.get('normalize_pixels', True),
                'augmentation': config.get('augmentation', False),
                'feature_extraction': config.get('feature_extraction', 'raw'),
                'color_mode': config.get('color_mode', 'rgb')
            }
            
            X_processed, _metadata = processor.process(X, y, processing_config)
            y_processed = y
            
            # Store processing history
            self.processing_history.append({
                'data_type': 'image',
                'original_shape': X.shape,
                'processed_shape': X_processed.shape,
                'config': processing_config,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            logger.info(f"✅ Image processing complete: {X.shape} → {X_processed.shape}")
            return X_processed, y_processed
            
        except Exception as e:
            logger.warning(f"⚠️ Image processing failed: {e}")
            return self._fallback_processing(X, y, 'image')
    
    def _process_time_series(self, X: pd.DataFrame, y: pd.Series, config: Dict[str, Any]) -> tuple:
        """Process time series data"""
        try:
            logger.info("🔧 Using TimeSeriesProcessor...")
            
            processor = TimeSeriesProcessor()
            
            # Setup processing configuration
            processing_config = {
                'window_size': config.get('window_size', 5),
                'difference': config.get('difference', True),
                'seasonal_decompose': config.get('seasonal_decompose', False),
                'lag_features': config.get('lag_features', True),
                'rolling_features': config.get('rolling_features', True),
                'scaling': config.get('scaling', True)
            }
            
            X_processed, _metadata = processor.process(X, y, processing_config)
            y_processed = y
            
            # Store processing history
            self.processing_history.append({
                'data_type': 'time_series',
                'original_shape': X.shape,
                'processed_shape': X_processed.shape,
                'config': processing_config,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            logger.info(f"✅ Time series processing complete: {X.shape} → {X_processed.shape}")
            return X_processed, y_processed
            
        except Exception as e:
            logger.warning(f"⚠️ Time series processing failed: {e}")
            return self._fallback_processing(X, y, 'time_series')
    
    def _fallback_processing(self, X: pd.DataFrame, y: pd.Series, data_type: str) -> tuple:
        """Fallback processing method"""
        try:
            logger.info(f"🔧 Using fallback processing for {data_type}...")
            
            X_processed = X.copy()
            y_processed = y.copy() if y is not None else y
            
            # Basic preprocessing
            # Handle missing values
            if X_processed.isnull().any().any():
                numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
                X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].mean())
                
                categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
                for col in categorical_cols:
                    X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0] if len(X_processed[col].mode()) > 0 else 'unknown')
            
            # Handle categorical variables
            categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                from sklearn.preprocessing import LabelEncoder
                for col in categorical_cols:
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            
            # Scale numeric features
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
            
            # Store processing history
            self.processing_history.append({
                'data_type': data_type,
                'original_shape': X.shape,
                'processed_shape': X_processed.shape,
                'method': 'fallback',
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            logger.info(f"✅ Fallback processing complete: {X.shape} → {X_processed.shape}")
            return X_processed, y_processed
            
        except Exception as e:
            logger.error(f"❌ Fallback processing failed: {e}")
            return X, y
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing activities"""
        return {
            'available_processors': self.available_processors,
            'total_processing_sessions': len(self.processing_history),
            'data_types_processed': list(set(p['data_type'] for p in self.processing_history)),
            'latest_processing': self.processing_history[-1] if self.processing_history else None
        }
    
    def generate_processing_report(self) -> str:
        """Generate processing report"""
        lines = []
        
        # Header
        lines.append("🔧 AutoForge Processor Report")
        lines.append("=" * 50)
        
        # Processor availability
        lines.append("\n📋 Available Processors:")
        for processor, available in self.available_processors.items():
            status = "✅" if available else "❌"
            lines.append(f"  {status} {processor}")
        
        # Processing history
        if self.processing_history:
            lines.append(f"\n🎯 Processing History:")
            lines.append(f"  Total Sessions: {len(self.processing_history)}")
            
            data_types = {}
            for proc in self.processing_history:
                dt = proc['data_type']
                data_types[dt] = data_types.get(dt, 0) + 1
            
            for data_type, count in data_types.items():
                lines.append(f"  {data_type}: {count} sessions")
        
        return "\n".join(lines)


# Global processor integrator instance
processor_integrator = ProcessorIntegrator()


def process_data_with_autoforge(X: pd.DataFrame, y: pd.Series = None, 
                              data_type: str = None, config: Dict[str, Any] = None) -> tuple:
    """Process data using AutoForge processors"""
    return processor_integrator.process_data(X, y, data_type, config)
