"""
Data-Type Intelligence System
Automatic detection and handling of time series and text data
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import re
from collections import Counter

logger = logging.getLogger(__name__)


class DataTypeDetector:
    """
    Intelligent data type detection for time series and text data
    """
    
    def __init__(self):
        """Initialize data type detector"""
        self.detection_rules = {
            "time_series": self._detect_time_series,
            "text": self._detect_text_data,
            "tabular": self._detect_tabular_data
        }
        
        self.detection_metadata = {}
    
    def detect_data_type(self, 
                        X: Union[np.ndarray, pd.DataFrame],
                        y: Optional[Union[np.ndarray, pd.Series]] = None,
                        column_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detects the primary data type and characteristics
        
        Args:
            X: Input features
            y: Target variable (optional)
            column_names: Column names (optional)
            
        Returns:
            Data type detection results
            
        Raises:
            ValueError: If input validation fails
            TypeError: If input types are incorrect
        """
        # Input validation
        if X is None:
            raise ValueError("X cannot be None")
        
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError("X must be numpy array or pandas DataFrame")
        
        if y is not None and not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError("y must be numpy array or pandas Series")
        
        if column_names is not None and not isinstance(column_names, list):
            raise TypeError("column_names must be a list or None")
        
        if isinstance(X, np.ndarray) and len(X.shape) < 2:
            raise ValueError("X must have at least 2 dimensions")
        
        logger.info("Analyzing data type characteristics...")
        
        # Convert to DataFrame for analysis
        if isinstance(X, np.ndarray):
            if column_names:
                df = pd.DataFrame(X, columns=column_names)
            else:
                df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            df = X.copy()
        
        # Basic data characteristics
        basic_info = {
            "n_samples": len(df),
            "n_features": df.shape[1],
            "has_target": y is not None,
            "data_types": {col: str(df[col].dtype) for col in df.columns}
        }
        
        # Detect primary data type
        detection_results = {}
        
        for data_type, detector_func in self.detection_rules.items():
            try:
                result = detector_func(df, y)
                detection_results[data_type] = result
            except Exception as e:
                logger.warning(f"Failed to detect {data_type}: {e}")
                detection_results[data_type] = {"detected": False, "confidence": 0.0, "error": str(e)}
        
        # Determine primary data type
        primary_type = self._determine_primary_type(detection_results)
        
        # Compile results
        results = {
            "primary_type": primary_type,
            "basic_info": basic_info,
            "detection_results": detection_results,
            "recommendations": self._get_recommendations(primary_type, detection_results),
            "processing_pipeline": self._get_processing_pipeline(primary_type, detection_results)
        }
        
        self.detection_metadata = results
        
        logger.info(f"Detected primary data type: {primary_type}")
        
        return results
    
    def _detect_time_series(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Detect if data is time series"""
        indicators = {
            "has_datetime_columns": False,
            "has_temporal_patterns": False,
            "has_sequential_structure": False,
            "has_seasonal_patterns": False,
            "confidence": 0.0
        }
        
        # Check for datetime columns
        datetime_cols = []
        for col in df.columns:
            if df[col].dtype in ['datetime64[ns]', 'datetime64']:
                datetime_cols.append(col)
                indicators["has_datetime_columns"] = True
            elif df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].head(100))
                    datetime_cols.append(col)
                    indicators["has_datetime_columns"] = True
                except:
                    pass
        
        # Check for temporal patterns in column names
        temporal_patterns = ['date', 'time', 'year', 'month', 'day', 'hour', 'minute', 'second']
        for col in df.columns:
            col_name = str(col).lower()
            if any(pattern in col_name for pattern in temporal_patterns):
                indicators["has_temporal_patterns"] = True
        
        # Check for sequential structure (if data looks ordered)
        if len(df) > 10:
            # Check if numeric columns show temporal patterns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Simple autocorrelation check
                for col in numeric_cols[:3]:  # Check first 3 numeric columns
                    series = df[col].dropna()
                    if len(series) > 10:
                        # Calculate simple autocorrelation
                        autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                        if not np.isnan(autocorr) and abs(autocorr) > 0.3:
                            indicators["has_sequential_structure"] = True
                            break
        
        # Check for seasonal patterns (if target is available)
        if y is not None and len(y) > 24:
            try:
                # Simple seasonality check using periodicity
                if len(y) >= 52:  # At least one year of weekly data
                    # Check for yearly seasonality
                    yearly_pattern = self._check_seasonality(y, period=52)
                    if yearly_pattern > 0.3:
                        indicators["has_seasonal_patterns"] = True
                
                if len(y) >= 12:  # At least one year of monthly data
                    # Check for monthly seasonality
                    monthly_pattern = self._check_seasonality(y, period=12)
                    if monthly_pattern > 0.3:
                        indicators["has_seasonal_patterns"] = True
                        
            except Exception:
                pass
        
        # Calculate confidence
        confidence = 0.0
        if indicators["has_datetime_columns"]:
            confidence += 0.4
        if indicators["has_temporal_patterns"]:
            confidence += 0.2
        if indicators["has_sequential_structure"]:
            confidence += 0.3
        if indicators["has_seasonal_patterns"]:
            confidence += 0.1
        
        indicators["confidence"] = min(confidence, 1.0)
        indicators["detected"] = indicators["confidence"] > 0.5
        indicators["datetime_columns"] = datetime_cols
        
        return indicators
    
    def _detect_text_data(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Detect if data contains text features"""
        indicators = {
            "has_text_columns": False,
            "has_high_cardinality": False,
            "has_natural_language": False,
            "confidence": 0.0
        }
        
        text_columns = []
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            # Skip if too few samples
            if len(col_data) < 10:
                continue
            
            # Check if it's object type (likely text)
            if df[col].dtype == 'object':
                # Check for high cardinality
                unique_ratio = col_data.nunique() / len(col_data)
                
                # Check average length of strings
                if len(col_data) > 0:
                    avg_length = col_data.astype(str).str.len().mean()
                    
                    # Check for natural language patterns
                    sample_text = col_data.astype(str).head(100)
                    text_features = self._analyze_text_features(sample_text)
                    
                    # Determine if this is text data
                    is_text = (
                        avg_length > 10 and  # Reasonable length
                        unique_ratio > 0.1 and  # Some variety
                        text_features["has_words"] and  # Contains words
                        text_features["word_diversity"] > 0.3  # Diverse vocabulary
                    )
                    
                    if is_text:
                        text_columns.append(col)
                        indicators["has_text_columns"] = True
                        
                        if unique_ratio > 0.5:
                            indicators["has_high_cardinality"] = True
                        
                        if text_features["has_words"]:
                            indicators["has_natural_language"] = True
        
        # Calculate confidence
        confidence = 0.0
        if indicators["has_text_columns"]:
            confidence += 0.4
        if indicators["has_high_cardinality"]:
            confidence += 0.3
        if indicators["has_natural_language"]:
            confidence += 0.3
        
        indicators["confidence"] = min(confidence, 1.0)
        indicators["detected"] = indicators["confidence"] > 0.5
        indicators["text_columns"] = text_columns
        
        return indicators
    
    def _detect_tabular_data(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Detect if data is standard tabular data"""
        indicators = {
            "has_numeric_features": False,
            "has_categorical_features": False,
            "structured_data": True,
            "confidence": 0.0
        }
        
        # Check for numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            indicators["has_numeric_features"] = True
        
        # Check for categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            indicators["has_categorical_features"] = True
        
        # Check if data is well-structured
        # (no missing values in critical positions, consistent types, etc.)
        missing_ratio = df.isnull().sum().sum() / df.size
        if missing_ratio < 0.5:  # Less than 50% missing
            indicators["structured_data"] = True
        else:
            indicators["structured_data"] = False
        
        # Calculate confidence
        confidence = 0.0
        if indicators["has_numeric_features"]:
            confidence += 0.4
        if indicators["has_categorical_features"]:
            confidence += 0.3
        if indicators["structured_data"]:
            confidence += 0.3
        
        indicators["confidence"] = min(confidence, 1.0)
        indicators["detected"] = indicators["confidence"] > 0.5
        
        return indicators
    
    def _check_seasonality(self, series: pd.Series, period: int) -> float:
        """Check for seasonality in a time series"""
        try:
            if len(series) < period * 2:
                return 0.0
            
            # Simple seasonality check using autocorrelation
            values = series.values
            autocorr = []
            
            for lag in range(1, min(period + 1, len(values) // 2)):
                if len(values) > lag:
                    corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorr.append(abs(corr))
            
            if autocorr:
                return max(autocorr)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _analyze_text_features(self, text_series: pd.Series) -> Dict[str, Any]:
        """Analyze text features to determine if it's natural language"""
        features = {
            "has_words": False,
            "word_diversity": 0.0,
            "avg_word_length": 0.0,
            "sentence_structure": False
        }
        
        try:
            all_text = " ".join(text_series.astype(str))
            
            # Check for words (letters separated by spaces)
            words = re.findall(r'\b[a-zA-Z]+\b', all_text)
            if len(words) > 0:
                features["has_words"] = True
                
                # Word diversity (unique words / total words)
                unique_words = set(word.lower() for word in words)
                features["word_diversity"] = len(unique_words) / len(words)
                
                # Average word length
                features["avg_word_length"] = sum(len(word) for word in words) / len(words)
                
                # Sentence structure (punctuation)
                if re.search(r'[.!?]', all_text):
                    features["sentence_structure"] = True
            
        except Exception:
            pass
        
        return features
    
    def _determine_primary_type(self, detection_results: Dict[str, Any]) -> str:
        """Determine the primary data type based on detection results"""
        # Get confidence scores
        confidences = {
            data_type: result.get("confidence", 0.0)
            for data_type, result in detection_results.items()
        }
        
        # Find the highest confidence
        primary_type = max(confidences, key=confidences.get)
        
        # Special case: if time series and text are both detected, prioritize time series
        if (detection_results.get("time_series", {}).get("detected", False) and 
            detection_results.get("text", {}).get("detected", False)):
            if confidences["time_series"] >= confidences["text"]:
                primary_type = "time_series"
            else:
                primary_type = "text"
        
        return primary_type
    
    def _get_recommendations(self, primary_type: str, detection_results: Dict[str, Any]) -> List[str]:
        """Get recommendations based on detected data type"""
        recommendations = []
        
        if primary_type == "time_series":
            recommendations.extend([
                "Consider using lag features for temporal dependencies",
                "Apply rolling window statistics for trend analysis",
                "Use seasonal decomposition for time series modeling",
                "Consider specialized time series models (ARIMA, Prophet)"
            ])
            
            if detection_results["time_series"].get("has_seasonal_patterns", False):
                recommendations.append("Seasonal patterns detected - consider seasonal models")
        
        elif primary_type == "text":
            recommendations.extend([
                "Apply TF-IDF vectorization for text features",
                "Consider text preprocessing (tokenization, stopword removal)",
                "Use text-specific models (Naive Bayes, Linear SVM)",
                "Consider word embeddings for semantic understanding"
            ])
            
            if detection_results["text"].get("has_high_cardinality", False):
                recommendations.append("High cardinality text detected - consider dimensionality reduction")
        
        elif primary_type == "tabular":
            recommendations.extend([
                "Standard preprocessing pipeline is appropriate",
                "Consider feature engineering for better performance",
                "Apply appropriate encoding for categorical variables"
            ])
        
        return recommendations
    
    def _get_processing_pipeline(self, primary_type: str, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommended processing pipeline"""
        pipeline = {
            "preprocessing_steps": [],
            "feature_engineering": [],
            "model_recommendations": []
        }
        
        if primary_type == "time_series":
            pipeline["preprocessing_steps"] = [
                "datetime_feature_extraction",
                "missing_value_imputation",
                "scaling"
            ]
            pipeline["feature_engineering"] = [
                "lag_features",
                "rolling_statistics",
                "seasonal_decomposition"
            ]
            pipeline["model_recommendations"] = [
                "time_series_forest",
                "gradient_boosting",
                "lstm_neural_network"
            ]
        
        elif primary_type == "text":
            pipeline["preprocessing_steps"] = [
                "text_cleaning",
                "tokenization",
                "vectorization"
            ]
            pipeline["feature_engineering"] = [
                "tfidf_features",
                "ngram_features",
                "text_embeddings"
            ]
            pipeline["model_recommendations"] = [
                "naive_bayes",
                "linear_svm",
                "gradient_boosting"
            ]
        
        elif primary_type == "tabular":
            pipeline["preprocessing_steps"] = [
                "missing_value_imputation",
                "categorical_encoding",
                "feature_scaling"
            ]
            pipeline["feature_engineering"] = [
                "polynomial_features",
                "interaction_features",
                "feature_selection"
            ]
            pipeline["model_recommendations"] = [
                "random_forest",
                "xgboost",
                "lightgbm"
            ]
        
        return pipeline


def detect_data_type(X: Union[np.ndarray, pd.DataFrame], 
                     y: Optional[Union[np.ndarray, pd.Series]] = None,
                     **kwargs) -> Dict[str, Any]:
    """Convenience function for data type detection"""
    detector = DataTypeDetector()
    return detector.detect_data_type(X, y, **kwargs)
