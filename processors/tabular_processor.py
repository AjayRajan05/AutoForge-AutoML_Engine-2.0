"""
📊 Tabular Data Processor
Specialized processor for tabular/structured data
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)


class TabularProcessor:
    """
    Tabular Data Processor
    
    Specialized processor for tabular/structured data with intelligent
    preprocessing, feature engineering, and optimization.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selectors = {}
        self.dimensionality_reducers = {}
        self.processing_history = []
        
    def process(self, X: pd.DataFrame, y: pd.Series = None, 
                config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process tabular data with intelligent preprocessing
        
        Args:
            X: Feature data
            y: Target data (optional)
            config: Processing configuration
            
        Returns:
            Processed data and processing metadata
            
        Raises:
            ValueError: If input validation fails
            TypeError: If input types are incorrect
        """
        # Input validation
        if X is None:
            raise ValueError("X cannot be None")
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError("X cannot be empty")
        
        if len(X.shape) < 2:
            raise ValueError("X must have at least 2 columns")
        
        if y is not None and not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be a pandas Series or numpy array")
        
        if config is not None and not isinstance(config, dict):
            raise TypeError("config must be a dictionary or None")
        
        logger.info("📊 Processing tabular data")
        start_time = time.time()
        
        config = config or self._get_default_config()
        if config.get("use_unified_pipeline", True):
            try:
                from execution.preprocessing_pipeline import PreprocessingPipeline
                task_type = config.get("task_type") or "classification"
                pipe = PreprocessingPipeline(
                    scale_features=config.get("scale_features", True),
                    feature_engineering=config.get("feature_engineering", False),
                )
                X_out, y_out, artifacts = pipe.fit_transform(
                    X, y if y is not None else pd.Series([0] * len(X)),
                    task_type=task_type,
                )
                metadata = {
                    "original_shape": X.shape,
                    "processed_shape": X_out.shape,
                    "processing_steps": ["unified_preprocessing_pipeline"],
                    "pipeline_type": "PreprocessingPipeline",
                    "artifacts": artifacts,
                }
                return X_out, metadata
            except Exception as exc:
                logger.warning("Unified pipeline failed, using legacy tabular path: %s", exc)

        metadata = {
            "original_shape": X.shape,
            "processing_steps": [],
            "feature_types": self._analyze_feature_types(X)
        }
        
        X_processed = X.copy()
        
        # Step 1: Handle missing values
        if config.get("handle_missing", True):
            X_processed, missing_metadata = self._handle_missing_values(X_processed, config)
            metadata["missing_handling"] = missing_metadata
            metadata["processing_steps"].append("missing_values")
        
        # Step 2: Handle outliers
        if config.get("handle_outliers", False):
            X_processed, outlier_metadata = self._handle_outliers(X_processed, config)
            metadata["outlier_handling"] = outlier_metadata
            metadata["processing_steps"].append("outliers")
        
        # Step 3: Encode categorical variables
        if config.get("encode_categorical", True):
            X_processed, encoding_metadata = self._encode_categorical(X_processed, config)
            metadata["categorical_encoding"] = encoding_metadata
            metadata["processing_steps"].append("categorical_encoding")
        
        # Step 4: Feature scaling
        if config.get("scale_features", True):
            X_processed, scaling_metadata = self._scale_features(X_processed, config)
            metadata["feature_scaling"] = scaling_metadata
            metadata["processing_steps"].append("feature_scaling")
        
        # Step 5: Feature engineering
        if config.get("feature_engineering", True):
            X_processed, engineering_metadata = self._engineer_features(X_processed, y, config)
            metadata["feature_engineering"] = engineering_metadata
            metadata["processing_steps"].append("feature_engineering")
        
        # Step 6: Feature selection
        if config.get("feature_selection", False) and y is not None:
            X_processed, selection_metadata = self._select_features(X_processed, y, config)
            metadata["feature_selection"] = selection_metadata
            metadata["processing_steps"].append("feature_selection")
        
        # Step 7: Dimensionality reduction
        if config.get("dimensionality_reduction", False):
            X_processed, reduction_metadata = self._reduce_dimensionality(X_processed, config, y)
            metadata["dimensionality_reduction"] = reduction_metadata
            metadata["processing_steps"].append("dimensionality_reduction")
        
        processing_time = time.time() - start_time
        
        metadata.update({
            "final_shape": X_processed.shape,
            "processing_time": processing_time,
            "features_removed": X.shape[1] - X_processed.shape[1],
            "features_added": X_processed.shape[1] - X.shape[1] + metadata.get("features_removed", 0)
        })
        
        # Store processing history
        self.processing_history.append(metadata)
        
        logger.info(f"✅ Tabular processing complete: {X.shape} → {X_processed.shape} in {processing_time:.2f}s")
        return X_processed, metadata
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default processing configuration"""
        return {
            "handle_missing": True,
            "missing_strategy": "adaptive",  # adaptive, mean, median, most_frequent, knn
            "handle_outliers": False,
            "outlier_method": "iqr",  # iqr, zscore, isolation_forest
            "encode_categorical": True,
            "encoding_method": "adaptive",  # adaptive, one_hot, label, target
            "scale_features": True,
            "scaling_method": "adaptive",  # adaptive, standard, minmax, robust
            "feature_engineering": True,
            "engineering_methods": ["polynomial", "interaction"],  # polynomial, interaction, clustering
            "feature_selection": False,
            "selection_method": "adaptive",  # adaptive, univariate, recursive, lasso
            "dimensionality_reduction": False,
            "reduction_method": "pca",  # pca, lda, ica
            "max_features": 1000
        }
    
    def _analyze_feature_types(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature types in the dataset"""
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_features = X.select_dtypes(include=['datetime64']).columns.tolist()
        boolean_features = X.select_dtypes(include=['bool']).columns.tolist()
        
        return {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "datetime_features": datetime_features,
            "boolean_features": boolean_features,
            "feature_counts": {
                "total": len(X.columns),
                "numeric": len(numeric_features),
                "categorical": len(categorical_features),
                "datetime": len(datetime_features),
                "boolean": len(boolean_features)
            }
        }
    
    def _handle_missing_values(self, X: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values with adaptive strategy"""
        logger.info("🔧 Handling missing values")
        
        strategy = config.get("missing_strategy", "adaptive")
        metadata = {"strategy": strategy, "features_processed": []}
        
        if strategy == "adaptive":
            strategy = self._select_missing_strategy(X)
        
        # Process each column
        for col in X.columns:
            if X[col].isnull().any():
                metadata["features_processed"].append(col)
                
                if X[col].dtype in ['object', 'category']:
                    # Categorical missing values
                    if strategy in ["mean", "median"]:
                        # Use most_frequent for categorical
                        X[col].fillna(X[col].mode()[0], inplace=True)
                        metadata["categorical_strategy"] = "most_frequent"
                    else:
                        X[col].fillna(X[col].mode()[0], inplace=True)
                else:
                    # Numeric missing values
                    if strategy == "mean":
                        X[col].fillna(X[col].mean(), inplace=True)
                    elif strategy == "median":
                        X[col].fillna(X[col].median(), inplace=True)
                    elif strategy == "most_frequent":
                        X[col].fillna(X[col].mode()[0], inplace=True)
                    elif strategy == "knn":
                        # KNN imputation
                        imputer = KNNImputer(n_neighbors=5)
                        X[[col]] = imputer.fit_transform(X[[col]])
                        self.imputers[col] = imputer
                    else:
                        X[col].fillna(X[col].mean(), inplace=True)
        
        metadata["missing_before"] = X.isnull().sum().sum()
        metadata["missing_after"] = 0
        
        return X, metadata
    
    def _select_missing_strategy(self, X: pd.DataFrame) -> str:
        """Select optimal missing value strategy based on data characteristics"""
        missing_ratio = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        
        if missing_ratio > 0.3:
            return "most_frequent"  # Conservative for high missing
        elif missing_ratio > 0.1:
            return "median"  # Robust for moderate missing
        else:
            return "mean"  # Efficient for low missing
    
    def _handle_outliers(self, X: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle outliers using specified method"""
        logger.info("🔧 Handling outliers")
        
        method = config.get("outlier_method", "iqr")
        metadata = {"method": method, "outliers_removed": 0}
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == "iqr":
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (X[col] < lower_bound) | (X[col] > upper_bound)
                metadata["outliers_removed"] += outliers.sum()
                
                # Cap outliers instead of removing
                X[col] = X[col].clip(lower_bound, upper_bound)
                
            elif method == "zscore":
                z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
                outliers = z_scores > 3
                metadata["outliers_removed"] += outliers.sum()
                
                # Cap outliers
                X[col] = np.where(outliers, X[col].median(), X[col])
        
        return X, metadata
    
    def _encode_categorical(self, X: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Encode categorical variables"""
        logger.info("🔧 Encoding categorical variables")
        
        method = config.get("encoding_method", "adaptive")
        metadata = {"method": method, "features_encoded": []}
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if method == "adaptive":
                encoding_method = self._select_encoding_method(X[col])
            else:
                encoding_method = method
            
            metadata["features_encoded"].append((col, encoding_method))
            
            if encoding_method == "one_hot":
                # One-hot encoding
                if X[col].nunique() < 20:  # Only for low cardinality
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
                    
            elif encoding_method == "label":
                # Label encoding
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
                
            elif encoding_method == "target":
                # Target encoding (simplified)
                if hasattr(self, 'target_encoding_map'):
                    # Use existing mapping
                    X[col] = X[col].map(self.target_encoding_map.get(col, {}))
                    X[col].fillna(0, inplace=True)
                # Note: Full target encoding would need y parameter
        
        metadata["final_categorical_count"] = len(X.select_dtypes(include=['object', 'category']).columns)
        
        return X, metadata
    
    def _select_encoding_method(self, series: pd.Series) -> str:
        """Select optimal encoding method for categorical series"""
        n_unique = series.nunique()
        
        if n_unique == 2:
            return "label"  # Binary
        elif n_unique < 10:
            return "one_hot"  # Low cardinality
        else:
            return "label"  # High cardinality
    
    def _scale_features(self, X: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scale numeric features"""
        logger.info("🔧 Scaling features")
        
        method = config.get("scaling_method", "adaptive")
        metadata = {"method": method, "features_scaled": []}
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if method == "adaptive":
            method = self._select_scaling_method(X[numeric_cols])
        
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            return X, metadata  # No scaling
        
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        self.scalers["global"] = scaler
        metadata["features_scaled"] = numeric_cols.tolist()
        
        return X, metadata
    
    def _select_scaling_method(self, X: pd.DataFrame) -> str:
        """Select optimal scaling method based on data characteristics"""
        # Check for outliers
        outlier_count = 0
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
            outlier_count += outliers
        
        outlier_ratio = outlier_count / (X.shape[0] * X.shape[1])
        
        if outlier_ratio > 0.05:
            return "robust"  # Use robust scaling for outliers
        else:
            return "standard"  # Use standard scaling
    
    def _engineer_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                          config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Engineer new features"""
        logger.info("🔧 Engineering features")
        
        methods = config.get("engineering_methods", ["polynomial"])
        metadata = {"methods": methods, "features_created": []}
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if "polynomial" in methods and len(numeric_cols) >= 2:
            # Create polynomial features (degree 2)
            for i, col1 in enumerate(numeric_cols[:5]):  # Limit to first 5 features
                for col2 in numeric_cols[i+1:i+6]:  # Limit combinations
                    if col1 != col2:
                        feature_name = f"{col1}_x_{col2}"
                        X[feature_name] = X[col1] * X[col2]
                        metadata["features_created"].append(feature_name)
        
        if "interaction" in methods and len(numeric_cols) >= 2:
            # Create interaction features
            for i, col1 in enumerate(numeric_cols[:3]):  # Limit to first 3 features
                for col2 in numeric_cols[i+1:i+4]:  # Limit combinations
                    if col1 != col2:
                        feature_name = f"{col1}_div_{col2}"
                        # Avoid division by zero
                        X[feature_name] = np.where(X[col2] != 0, X[col1] / X[col2], 0)
                        metadata["features_created"].append(feature_name)
        
        if "statistical" in methods:
            # Create statistical features
            for col in numeric_cols[:5]:  # Limit to first 5 features
                # Rolling statistics would require time series data
                # Instead, create ratio features
                if X[col].mean() != 0:
                    feature_name = f"{col}_ratio_to_mean"
                    X[feature_name] = X[col] / X[col].mean()
                    metadata["features_created"].append(feature_name)
        
        return X, metadata
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, 
                        config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Select best features"""
        logger.info("🔧 Selecting features")
        
        method = config.get("selection_method", "adaptive")
        metadata = {"method": method, "features_selected": [], "features_removed": []}
        
        if method == "adaptive":
            method = self._select_feature_selection_method(X, y)
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if method == "univariate":
            # Univariate feature selection
            if len(y.unique()) < 20:  # Classification
                selector = SelectKBest(f_classif, k=min(50, len(numeric_cols)))
            else:  # Regression
                selector = SelectKBest(f_regression, k=min(50, len(numeric_cols)))
            
            X_selected = selector.fit_transform(X[numeric_cols], y)
            selected_features = [numeric_cols[i] for i in selector.get_support(indices=True)]
            
            X = pd.DataFrame(X_selected, columns=selected_features)
            self.feature_selectors["univariate"] = selector
            
        elif method == "recursive":
            # Recursive feature elimination
            if len(y.unique()) < 20:  # Classification
                estimator = RandomForestClassifier(n_estimators=config.get('n_estimators', 50), random_state=42)
            else:  # Regression
                estimator = RandomForestRegressor(n_estimators=config.get('n_estimators', 50), random_state=42)
            
            selector = RFE(estimator, n_features_to_select=min(30, len(numeric_cols)))
            X_selected = selector.fit_transform(X[numeric_cols], y)
            selected_features = [numeric_cols[i] for i in selector.get_support(indices=True)]
            
            X = pd.DataFrame(X_selected, columns=selected_features)
            self.feature_selectors["recursive"] = selector
        
        metadata["features_selected"] = selected_features
        metadata["features_removed"] = [col for col in numeric_cols if col not in selected_features]
        
        return X, metadata
    
    def _select_feature_selection_method(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Select optimal feature selection method"""
        n_samples, n_features = X.shape
        
        if n_features > 100:
            return "univariate"  # Faster for high dimensional data
        elif n_samples < 1000:
            return "recursive"  # Better for small datasets
        else:
            return "univariate"  # Balanced approach
    
    def _reduce_dimensionality(self, X: pd.DataFrame, config: Dict[str, Any],
                             y: pd.Series = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Reduce dimensionality"""
        logger.info("🔧 Reducing dimensionality")
        
        method = config.get("reduction_method", "pca")
        metadata = {"method": method, "original_features": X.shape[1]}
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if method == "pca":
            # Principal Component Analysis
            n_components = min(50, len(numeric_cols), X.shape[0] - 1)
            pca = PCA(n_components=n_components)
            
            X_reduced = pca.fit_transform(X[numeric_cols])
            
            # Create column names
            pca_columns = [f"PC_{i+1}" for i in range(n_components)]
            X = pd.DataFrame(X_reduced, columns=pca_columns)
            
            self.dimensionality_reducers["pca"] = pca
            metadata["explained_variance_ratio"] = pca.explained_variance_ratio_.sum()
        
        metadata["final_features"] = X.shape[1]
        
        return X, metadata
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of all processing operations"""
        if not self.processing_history:
            return {"error": "No processing history available"}
        
        latest = self.processing_history[-1]
        
        return {
            "total_processing_sessions": len(self.processing_history),
            "latest_session": {
                "original_shape": latest["original_shape"],
                "final_shape": latest["final_shape"],
                "processing_time": latest["processing_time"],
                "steps_applied": latest["processing_steps"],
                "features_removed": latest.get("features_removed", 0),
                "features_added": latest.get("features_added", 0)
            },
            "feature_types": latest.get("feature_types", {}),
            "all_steps_applied": list(set(step for session in self.processing_history for step in session["processing_steps"]))
        }
    
    def transform_new_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted processors"""
        X_processed = X.copy()
        
        # Apply missing value handling
        for col, imputer in self.imputers.items():
            if col in X_processed.columns:
                X_processed[[col]] = imputer.transform(X_processed[[col]])
        
        # Apply categorical encoding
        for col, encoder in self.encoders.items():
            if col in X_processed.columns:
                X_processed[col] = encoder.transform(X_processed[col].astype(str))
        
        # Apply scaling
        if "global" in self.scalers:
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            X_processed[numeric_cols] = self.scalers["global"].transform(X_processed[numeric_cols])
        
        return X_processed
    
    def reset(self):
        """Reset all fitted processors"""
        self.scalers.clear()
        self.encoders.clear()
        self.imputers.clear()
        self.feature_selectors.clear()
        self.dimensionality_reducers.clear()
        self.processing_history.clear()
        logger.info("🔄 Tabular processor reset")
