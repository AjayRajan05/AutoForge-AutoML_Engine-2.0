"""
🧠 Dataset Intelligence Analyzer
Analyzes dataset characteristics to inform strategy selection
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)


class DatasetIntelligence:
    """
    Dataset Intelligence Analyzer
    
    Analyzes datasets to extract meaningful characteristics that guide
    AutoML strategy selection and optimization decisions.
    """
    
    def __init__(self):
        self.analysis_cache = {}
        
    def analyze(self, X: Union[pd.DataFrame, np.ndarray], 
               y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Comprehensive dataset analysis
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Dictionary containing dataset characteristics
            
        Raises:
            ValueError: If input validation fails
            TypeError: If input types are incorrect
        """
        # Input validation
        if X is None:
            raise ValueError("X cannot be None")
        
        if y is None:
            raise ValueError("y cannot be None")
        
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be pandas DataFrame or numpy array")
        
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be pandas Series or numpy array")
        
        if hasattr(X, 'empty') and X.empty:
            raise ValueError("X cannot be empty")
        
        if hasattr(y, 'empty') and y.empty:
            raise ValueError("y cannot be empty")
        
        try:
            # Convert to DataFrame for easier analysis
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            
            if not isinstance(y, (pd.Series, np.ndarray)):
                y = pd.Series(y)
            
            # Generate unique cache key
            cache_key = self._generate_cache_key(X, y)
            
            # Check cache
            if cache_key in self.analysis_cache:
                logger.info("📋 Using cached dataset analysis")
                return self.analysis_cache[cache_key]
            
            logger.info("🔍 Analyzing dataset characteristics...")
            
            analysis = {
                # Size characteristics
                "size_profile": self._analyze_size(X),
                "shape_info": self._get_shape_info(X),
                
                # Quality characteristics
                "quality_profile": self._analyze_quality(X),
                "missing_profile": self._analyze_missing(X),
                
                # Type characteristics
                "type_profile": self._analyze_types(X),
                "feature_types": self._get_feature_types(X),
                
                # Complexity characteristics
                "complexity_profile": self._analyze_complexity(X, y),
                "target_profile": self._analyze_target(y),
                
                # Statistical characteristics
                "statistical_profile": self._analyze_statistics(X),
                "correlation_profile": self._analyze_correlations(X),
                
                # Recommendations
                "recommendations": self._generate_recommendations(X, y)
            }
            
            # Cache results
            self.analysis_cache[cache_key] = analysis
            
            logger.info(f"✅ Dataset analysis complete: {len(analysis)} characteristics")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Dataset analysis failed: {e}")
            return self._get_fallback_analysis(X, y)
    
    def _analyze_size(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset size characteristics"""
        n_samples, n_features = X.shape
        
        # Size classification
        # Use configurable thresholds from settings
        try:
            from config.settings import get_config_value
            small_threshold = get_config_value('dataset_analysis', 'small_dataset_threshold', 1000)
            medium_threshold = get_config_value('dataset_analysis', 'medium_dataset_threshold', 100000)
            high_dim_threshold = get_config_value('dataset_analysis', 'high_dimensional_threshold', 1000)
        except ImportError:
            # Fallback to hardcoded values if config not available
            small_threshold = 1000
            medium_threshold = 100000
            high_dim_threshold = 1000
        
        if n_samples < small_threshold:
            size_category = "small"
            description = f"Small dataset (< {small_threshold:,} samples)"
        elif n_samples < medium_threshold:
            size_category = "medium"
            description = f"Medium dataset ({small_threshold:,}-{medium_threshold:,} samples)"
        else:
            size_category = "large"
            description = f"Large dataset (> {medium_threshold:,} samples)"
        
        # Feature density
        feature_density = n_features / n_samples if n_samples > 0 else 0
        
        return {
            "category": size_category,
            "description": description,
            "n_samples": n_samples,
            "n_features": n_features,
            "feature_density": feature_density,
            "sample_feature_ratio": n_samples / n_features if n_features > 0 else 0,
            "memory_estimate_mb": self._estimate_memory_usage(X),
            "is_high_dimensional": n_features > high_dim_threshold,
            "is_wide": n_features > n_samples
        }
    
    def _analyze_quality(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality characteristics"""
        # Missing value analysis
        total_cells = X.shape[0] * X.shape[1]
        missing_cells = X.isnull().sum().sum()
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
        
        # Quality classification
        if missing_ratio > 0.3:
            quality_category = "poor"
            description = "Poor quality (>30% missing)"
        elif missing_ratio > 0.1:
            quality_category = "moderate"
            description = "Moderate quality (10-30% missing)"
        else:
            quality_category = "good"
            description = "Good quality (<10% missing)"
        
        # Duplicate analysis
        duplicate_rows = X.duplicated().sum()
        duplicate_ratio = duplicate_rows / len(X) if len(X) > 0 else 0
        
        return {
            "category": quality_category,
            "description": description,
            "missing_ratio": missing_ratio,
            "missing_cells": missing_cells,
            "total_cells": total_cells,
            "duplicate_ratio": duplicate_ratio,
            "duplicate_rows": int(duplicate_rows),
            "has_missing": missing_ratio > 0,
            "has_duplicates": duplicate_ratio > 0
        }
    
    def _analyze_missing(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Detailed missing value analysis"""
        missing_by_column = X.isnull().sum()
        missing_ratios = missing_by_column / len(X)
        
        # Missing patterns
        completely_missing = (missing_ratios == 1.0).sum()
        mostly_missing = ((missing_ratios > 0.5) & (missing_ratios < 1.0)).sum()
        partially_missing = ((missing_ratios > 0) & (missing_ratios <= 0.5)).sum()
        
        return {
            "columns_with_missing": (missing_ratios > 0).sum(),
            "completely_missing_columns": int(completely_missing),
            "mostly_missing_columns": int(mostly_missing),
            "partially_missing_columns": int(partially_missing),
            "max_missing_ratio": missing_ratios.max() if len(missing_ratios) > 0 else 0,
            "avg_missing_ratio": missing_ratios.mean() if len(missing_ratios) > 0 else 0,
            "missing_pattern": "random" if missing_ratios.std() < 0.2 else "structured",
            "problematic_columns": missing_ratios[missing_ratios > 0.5].index.tolist()
        }
    
    def _analyze_types(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data type characteristics"""
        dtype_counts = X.dtypes.value_counts()
        
        # Categorize types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = X.select_dtypes(include=['datetime64']).columns.tolist()
        boolean_cols = X.select_dtypes(include=['bool']).columns.tolist()
        
        # Check for mixed types in object columns
        mixed_type_cols = []
        for col in categorical_cols:
            unique_types = set(type(val).__name__ for val in X[col].dropna().head(100))
            if len(unique_types) > 1:
                mixed_type_cols.append(col)
        
        return {
            "dtype_distribution": dtype_counts.to_dict(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "boolean_columns": boolean_cols,
            "mixed_type_columns": mixed_type_cols,
            "has_categorical": len(categorical_cols) > 0,
            "has_numeric": len(numeric_cols) > 0,
            "has_datetime": len(datetime_cols) > 0,
            "has_mixed_types": len(mixed_type_cols) > 0,
            "categorical_ratio": len(categorical_cols) / len(X.columns) if len(X.columns) > 0 else 0
        }
    
    def _analyze_complexity(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Analyze dataset complexity"""
        # Target complexity
        if len(y.unique()) < 20:
            task_type = "classification"
            n_classes = len(y.unique())
            class_balance = y.value_counts().min() / y.value_counts().max()
            is_imbalanced = class_balance < 0.5
        else:
            task_type = "regression"
            n_classes = 1
            class_balance = 1.0
            is_imbalanced = False
        
        # Feature complexity
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        # Sparsity (for sparse data)
        if hasattr(X, 'values'):
            zero_ratio = (X.values == 0).mean()
        else:
            zero_ratio = 0
        
        # Dimensionality stress
        dimensionality_stress = n_features / n_samples if n_samples > 0 else 0
        
        return {
            "task_type": task_type,
            "n_classes": n_classes,
            "class_balance": class_balance,
            "is_imbalanced": is_imbalanced,
            "is_multiclass": task_type == "classification" and n_classes > 2,
            "sparsity": zero_ratio,
            "dimensionality_stress": dimensionality_stress,
            "is_high_stress": dimensionality_stress > 0.1,
            "complexity_score": self._calculate_complexity_score(X, y)
        }
    
    def _analyze_target(self, y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Analyze target variable characteristics"""
        if len(y.unique()) < 20:
            # Classification target
            value_counts = y.value_counts()
            return {
                "type": "classification",
                "n_classes": len(y.unique()),
                "class_distribution": value_counts.to_dict(),
                "majority_class": value_counts.index[0],
                "majority_ratio": value_counts.iloc[0] / len(y),
                "minority_class": value_counts.index[-1],
                "minority_ratio": value_counts.iloc[-1] / len(y),
                "is_binary": len(y.unique()) == 2,
                "is_multiclass": len(y.unique()) > 2
            }
        else:
            # Regression target
            return {
                "type": "regression",
                "min": float(y.min()),
                "max": float(y.max()),
                "mean": float(y.mean()),
                "std": float(y.std()),
                "skewness": float(y.skew()) if hasattr(y, 'skew') else 0,
                "range": float(y.max() - y.min()),
                "has_outliers": self._has_outliers(y)
            }
    
    def _analyze_statistics(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze statistical characteristics"""
        numeric_cols = X.select_dtypes(include=[np.number])
        
        if len(numeric_cols.columns) == 0:
            return {"error": "No numeric columns for statistical analysis"}
        
        stats = {}
        for col in numeric_cols.columns:
            col_data = numeric_cols[col].dropna()
            if len(col_data) > 0:
                stats[col] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "median": float(col_data.median()),
                    "skewness": float(col_data.skew()),
                    "kurtosis": float(col_data.kurtosis()),
                    "has_outliers": self._has_outliers(col_data)
                }
        
        return {
            "column_stats": stats,
            "overall_skewness": np.mean([s["skewness"] for s in stats.values()]),
            "overall_kurtosis": np.mean([s["kurtosis"] for s in stats.values()]),
            "high_skewness_columns": [col for col, s in stats.items() if abs(s["skewness"]) > 2],
            "outlier_columns": [col for col, s in stats.items() if s["has_outliers"]]
        }
    
    def _analyze_correlations(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation patterns"""
        numeric_cols = X.select_dtypes(include=[np.number])
        
        if len(numeric_cols.columns) < 2:
            return {"error": "Need at least 2 numeric columns for correlation analysis"}
        
        correlation_matrix = numeric_cols.corr()
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append({
                        "feature1": correlation_matrix.columns[i],
                        "feature2": correlation_matrix.columns[j],
                        "correlation": float(corr_val)
                    })
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "high_correlation_pairs": high_corr_pairs,
            "max_correlation": float(correlation_matrix.abs().max().max()),
            "avg_correlation": float(correlation_matrix.abs().mean().mean()),
            "has_high_correlations": len(high_corr_pairs) > 0
        }
    
    def _get_shape_info(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Get basic shape information"""
        return {
            "n_rows": X.shape[0],
            "n_columns": X.shape[1],
            "is_square": X.shape[0] == X.shape[1],
            "is_tall": X.shape[0] > X.shape[1],
            "is_wide": X.shape[1] > X.shape[0]
        }
    
    def _get_feature_types(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed feature type information"""
        feature_info = {}
        
        for col in X.columns:
            dtype = str(X[col].dtype)
            n_unique = X[col].nunique()
            
            # Classify feature type
            if dtype in ['int64', 'float64']:
                if n_unique == 2 and X[col].min() == 0 and X[col].max() == 1:
                    feature_type = "binary"
                elif n_unique < 20:
                    feature_type = "ordinal"
                else:
                    feature_type = "continuous"
            elif dtype == 'object':
                if n_unique < 20:
                    feature_type = "categorical"
                else:
                    feature_type = "text"
            elif dtype == 'bool':
                feature_type = "boolean"
            elif 'datetime' in dtype:
                feature_type = "datetime"
            else:
                feature_type = "unknown"
            
            feature_info[col] = {
                "dtype": dtype,
                "feature_type": feature_type,
                "n_unique": int(n_unique),
                "unique_ratio": n_unique / len(X),
                "has_missing": X[col].isnull().any()
            }
        
        return feature_info
    
    def _generate_recommendations(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Generate recommendations based on analysis"""
        recommendations = {
            "preprocessing": [],
            "feature_engineering": [],
            "model_selection": [],
            "optimization": []
        }
        
        # Size-based recommendations
        if len(X) > 100000:
            recommendations["preprocessing"].append("sampling")
            recommendations["optimization"].append("early_stopping")
        
        # Quality-based recommendations
        missing_ratio = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        if missing_ratio > 0.2:
            recommendations["preprocessing"].append("robust_imputation")
        elif missing_ratio > 0:
            recommendations["preprocessing"].append("simple_imputation")
        
        # Type-based recommendations
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            recommendations["preprocessing"].append("categorical_encoding")
        
        # Complexity-based recommendations
        if len(y.unique()) > 2:
            recommendations["model_selection"].append("multiclass_capable")
        
        # Correlation-based recommendations
        numeric_cols = X.select_dtypes(include=[np.number])
        if len(numeric_cols.columns) > 1:
            corr_matrix = numeric_cols.corr()
            if (corr_matrix.abs() > 0.8).any().any():
                recommendations["feature_engineering"].append("feature_selection")
        
        return recommendations
    
    def _calculate_complexity_score(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate overall dataset complexity score (0-1)"""
        n_samples, n_features = X.shape
        n_classes = len(y.unique())
        
        # Normalize factors
        size_factor = min(n_samples / 10000, 1.0)  # Normalize to 10K samples
        feature_factor = min(n_features / 100, 1.0)   # Normalize to 100 features
        class_factor = min(n_classes / 10, 1.0)       # Normalize to 10 classes
        
        # Combine factors
        complexity = (size_factor + feature_factor + class_factor) / 3
        
        return float(complexity)
    
    def _has_outliers(self, data: Union[pd.Series, np.ndarray]) -> bool:
        """Check if data has outliers using IQR method"""
        if len(data) < 4:
            return False
        
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (data < lower_bound) | (data > upper_bound)
        return outliers.any()
    
    def _estimate_memory_usage(self, X: pd.DataFrame) -> float:
        """Estimate memory usage in MB"""
        memory_bytes = X.memory_usage(deep=True).sum()
        return memory_bytes / (1024 * 1024)  # Convert to MB
    
    def _generate_cache_key(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> str:
        """Generate cache key for dataset"""
        dtypes_str = "|".join(str(dt) for dt in X.dtypes.tolist())
        return f"{X.shape}_{dtypes_str}_{len(pd.Series(y).unique())}"
    
    def _get_fallback_analysis(self, X: Union[pd.DataFrame, np.ndarray], 
                              y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Fallback analysis when main analysis fails"""
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if len(y.unique()) < 20:
            task_type = "classification"
        elif pd.api.types.is_numeric_dtype(y):
            task_type = "regression"
        else:
            task_type = "classification"
        return {
            "size_profile": {"category": "unknown", "n_samples": len(X), "n_features": X.shape[1] if hasattr(X, 'shape') else 1},
            "quality_profile": {"category": "unknown", "missing_ratio": 0},
            "type_profile": {"category": "unknown"},
            "complexity_profile": {"task_type": task_type},
            "error": "Analysis failed, using fallback"
        }
