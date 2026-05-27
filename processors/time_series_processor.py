"""
📈 Time Series Data Processor
Specialized processor for time series data with temporal features
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.signal import find_peaks
import warnings

logger = logging.getLogger(__name__)


class TimeSeriesProcessor:
    """
    Time Series Data Processor
    
    Specialized processor for time series data with intelligent:
    - Temporal feature engineering
    - Seasonality detection
    - Trend analysis
    - Lag feature creation
    - Rolling window statistics
    """
    
    def __init__(self):
        self.scalers = {}
        self.seasonality_info = {}
        self.trend_info = {}
        self.feature_importance = {}
        self.processing_history = []
        
    def process(self, X: pd.DataFrame, y: pd.Series = None, 
                config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process time series data with intelligent temporal feature engineering
        
        Args:
            X: Feature data (should include datetime column)
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
        
        logger.info("📈 Processing time series data")
        start_time = time.time()
        
        config = config or self._get_default_config()
        metadata = {
            "original_shape": X.shape,
            "processing_steps": [],
            "time_features": []
        }
        
        X_processed = X.copy()
        
        # Step 1: Detect and validate time series structure
        datetime_col = self._detect_datetime_column(X_processed)
        if datetime_col is None:
            raise ValueError("No datetime column found in time series data")
        
        metadata["datetime_column"] = datetime_col
        X_processed[datetime_col] = pd.to_datetime(X_processed[datetime_col])
        X_processed = X_processed.sort_values(datetime_col)
        
        # Step 2: Extract temporal features
        if config.get("extract_temporal_features", True):
            X_processed, temporal_metadata = self._extract_temporal_features(
                X_processed, datetime_col, config
            )
            metadata["temporal_features"] = temporal_metadata
            metadata["processing_steps"].append("temporal_features")
        
        # Step 3: Create lag features
        if config.get("create_lag_features", True):
            X_processed, lag_metadata = self._create_lag_features(
                X_processed, datetime_col, config
            )
            metadata["lag_features"] = lag_metadata
            metadata["processing_steps"].append("lag_features")
        
        # Step 4: Create rolling window features
        if config.get("create_rolling_features", True):
            X_processed, rolling_metadata = self._create_rolling_features(
                X_processed, datetime_col, config
            )
            metadata["rolling_features"] = rolling_metadata
            metadata["processing_steps"].append("rolling_features")
        
        # Step 5: Detect and handle seasonality
        if config.get("detect_seasonality", True):
            X_processed, seasonality_metadata = self._detect_and_handle_seasonality(
                X_processed, datetime_col, config
            )
            metadata["seasonality"] = seasonality_metadata
            metadata["processing_steps"].append("seasonality_detection")
        
        # Step 6: Detect and handle trends
        if config.get("detect_trends", True):
            X_processed, trend_metadata = self._detect_and_handle_trends(
                X_processed, datetime_col, config
            )
            metadata["trends"] = trend_metadata
            metadata["processing_steps"].append("trend_detection")
        
        # Step 7: Create difference features
        if config.get("create_difference_features", True):
            X_processed, diff_metadata = self._create_difference_features(
                X_processed, datetime_col, config
            )
            metadata["difference_features"] = diff_metadata
            metadata["processing_steps"].append("difference_features")
        
        # Step 8: Handle missing values (time series specific)
        if config.get("handle_missing", True):
            X_processed, missing_metadata = self._handle_time_series_missing(
                X_processed, datetime_col, config
            )
            metadata["missing_handling"] = missing_metadata
            metadata["processing_steps"].append("missing_values")
        
        # Step 9: Scale features
        if config.get("scale_features", True):
            X_processed, scaling_metadata = self._scale_time_series_features(
                X_processed, datetime_col, config
            )
            metadata["feature_scaling"] = scaling_metadata
            metadata["processing_steps"].append("feature_scaling")
        
        # Step 10: Remove datetime column (optional)
        if config.get("drop_datetime", True):
            X_processed = X_processed.drop(columns=[datetime_col])
        
        processing_time = time.time() - start_time
        
        metadata.update({
            "final_shape": X_processed.shape,
            "processing_time": processing_time,
            "features_created": X_processed.shape[1] - X.shape[1]
        })
        
        # Store processing history
        self.processing_history.append(metadata)
        
        logger.info(f"✅ Time series processing complete: {X.shape} → {X_processed.shape} in {processing_time:.2f}s")
        return X_processed, metadata
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default time series processing configuration"""
        return {
            "extract_temporal_features": True,
            "create_lag_features": True,
            "create_rolling_features": True,
            "detect_seasonality": True,
            "detect_trends": True,
            "create_difference_features": True,
            "handle_missing": True,
            "scale_features": True,
            "drop_datetime": False,
            
            # Feature engineering parameters
            "max_lag_periods": config.get("max_lag_periods", 12),
            "rolling_windows": [3, 7, 14, 30],
            "seasonality_periods": config.get("seasonality_periods", [7, 30, 365]),  # Weekly, monthly, yearly
            "difference_periods": config.get("difference_periods", [1, 7, 30]),
            
            # Missing value handling
            "missing_strategy": "interpolate",  # interpolate, forward_fill, backward_fill
            
            # Scaling
            "scaling_method": "standard"  # standard, minmax, robust
        }
    
    def _detect_datetime_column(self, X: pd.DataFrame) -> Optional[str]:
        """Detect datetime column in the dataset"""
        for col in X.columns:
            if X[col].dtype == 'datetime64[ns]':
                return col
            elif X[col].dtype == 'object':
                try:
                    pd.to_datetime(X[col].head())
                    return col
                except:
                    continue
        return None
    
    def _extract_temporal_features(self, X: pd.DataFrame, datetime_col: str,
                                 config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extract temporal features from datetime column"""
        logger.info("📅 Extracting temporal features")
        
        datetime_series = X[datetime_col]
        features = {}
        
        # Basic temporal features
        features['year'] = datetime_series.dt.year
        features['month'] = datetime_series.dt.month
        features['day'] = datetime_series.dt.day
        features['dayofweek'] = datetime_series.dt.dayofweek
        features['dayofyear'] = datetime_series.dt.dayofyear
        features['hour'] = datetime_series.dt.hour
        features['minute'] = datetime_series.dt.minute
        features['quarter'] = datetime_series.dt.quarter
        features['week'] = datetime_series.dt.isocalendar().week
        features['is_weekend'] = (datetime_series.dt.dayofweek >= 5).astype(int)
        features['is_month_start'] = datetime_series.dt.is_month_start.astype(int)
        features['is_month_end'] = datetime_series.dt.is_month_end.astype(int)
        features['is_quarter_start'] = datetime_series.dt.is_quarter_start.astype(int)
        features['is_quarter_end'] = datetime_series.dt.is_quarter_end.astype(int)
        
        # Cyclical encoding for temporal features
        features['month_sin'] = np.sin(2 * np.pi * datetime_series.dt.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * datetime_series.dt.month / 12)
        features['day_sin'] = np.sin(2 * np.pi * datetime_series.dt.day / 31)
        features['day_cos'] = np.cos(2 * np.pi * datetime_series.dt.day / 31)
        features['hour_sin'] = np.sin(2 * np.pi * datetime_series.dt.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * datetime_series.dt.hour / 24)
        features['dayofweek_sin'] = np.sin(2 * np.pi * datetime_series.dt.dayofweek / 7)
        features['dayofweek_cos'] = np.cos(2 * np.pi * datetime_series.dt.dayofweek / 7)
        
        # Add features to dataframe
        for feature_name, feature_data in features.items():
            X[feature_name] = feature_data
        
        metadata = {
            "features_created": list(features.keys()),
            "temporal_range": {
                "start": datetime_series.min(),
                "end": datetime_series.max(),
                "duration_days": (datetime_series.max() - datetime_series.min()).days
            }
        }
        
        return X, metadata
    
    def _create_lag_features(self, X: pd.DataFrame, datetime_col: str,
                           config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create lag features for time series"""
        logger.info("⏰ Creating lag features")
        
        max_lag = config.get("max_lag_periods", 12)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        lag_features = {}
        
        for col in numeric_cols:
            for lag in range(1, max_lag + 1):
                lag_feature_name = f"{col}_lag_{lag}"
                lag_features[lag_feature_name] = X[col].shift(lag)
        
        # Add lag features to dataframe
        for feature_name, feature_data in lag_features.items():
            X[feature_name] = feature_data
        
        metadata = {
            "lag_features_created": list(lag_features.keys()),
            "max_lag_period": max_lag,
            "numeric_columns_used": list(numeric_cols)
        }
        
        return X, metadata
    
    def _create_rolling_features(self, X: pd.DataFrame, datetime_col: str,
                               config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create rolling window features"""
        logger.info("📊 Creating rolling window features")
        
        windows = config.get("rolling_windows", [3, 7, 14, 30])
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        rolling_features = {}
        
        for col in numeric_cols:
            for window in windows:
                # Rolling mean
                rolling_mean_name = f"{col}_rolling_mean_{window}"
                rolling_features[rolling_mean_name] = X[col].rolling(window=window).mean()
                
                # Rolling std
                rolling_std_name = f"{col}_rolling_std_{window}"
                rolling_features[rolling_std_name] = X[col].rolling(window=window).std()
                
                # Rolling min/max
                rolling_min_name = f"{col}_rolling_min_{window}"
                rolling_features[rolling_min_name] = X[col].rolling(window=window).min()
                
                rolling_max_name = f"{col}_rolling_max_{window}"
                rolling_features[rolling_max_name] = X[col].rolling(window=window).max()
                
                # Rolling median
                rolling_median_name = f"{col}_rolling_median_{window}"
                rolling_features[rolling_median_name] = X[col].rolling(window=window).median()
        
        # Add rolling features to dataframe
        for feature_name, feature_data in rolling_features.items():
            X[feature_name] = feature_data
        
        metadata = {
            "rolling_features_created": list(rolling_features.keys()),
            "windows_used": windows,
            "numeric_columns_used": list(numeric_cols)
        }
        
        return X, metadata
    
    def _detect_and_handle_seasonality(self, X: pd.DataFrame, datetime_col: str,
                                      config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect and handle seasonality in the data"""
        logger.info("🔄 Detecting seasonality")
        
        periods = config.get("seasonality_periods", [7, 30, 365])
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        seasonality_info = {}
        
        for col in numeric_cols:
            col_seasonality = {}
            
            for period in periods:
                if len(X[col]) >= period * 2:  # Need at least 2 cycles
                    try:
                        # Use autocorrelation to detect seasonality
                        autocorr = []
                        for lag in range(1, period + 1):
                            corr = X[col].autocorr(lag=lag)
                            if not np.isnan(corr):
                                autocorr.append(abs(corr))
                        
                        if autocorr:
                            max_autocorr = max(autocorr)
                            if max_autocorr > 0.3:  # Threshold for seasonality
                                col_seasonality[period] = {
                                    "detected": True,
                                    "max_autocorr": max_autocorr,
                                    "strength": "strong" if max_autocorr > 0.7 else "moderate" if max_autocorr > 0.5 else "weak"
                                }
                            else:
                                col_seasonality[period] = {"detected": False}
                        
                    except Exception as e:
                        logger.warning(f"Failed to detect seasonality for {col} period {period}: {e}")
                        col_seasonality[period] = {"detected": False, "error": str(e)}
            
            if col_seasonality:
                seasonality_info[col] = col_seasonality
        
        # Create seasonal features if seasonality detected
        seasonal_features = {}
        
        for col, col_seasonality in seasonality_info.items():
            for period, info in col_seasonality.items():
                if info.get("detected", False):
                    # Create seasonal lag features
                    for lag in [period, period * 2]:
                        feature_name = f"{col}_seasonal_lag_{lag}"
                        seasonal_features[feature_name] = X[col].shift(lag)
        
        # Add seasonal features to dataframe
        for feature_name, feature_data in seasonal_features.items():
            X[feature_name] = feature_data
        
        metadata = {
            "seasonality_detected": seasonality_info,
            "seasonal_features_created": list(seasonal_features.keys()),
            "periods_tested": periods
        }
        
        self.seasonality_info = seasonality_info
        
        return X, metadata
    
    def _detect_and_handle_trends(self, X: pd.DataFrame, datetime_col: str,
                                config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect and handle trends in the data"""
        logger.info("📈 Detecting trends")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        trend_info = {}
        
        for col in numeric_cols:
            try:
                # Use linear regression to detect trend
                x = np.arange(len(X[col]))
                y = X[col].dropna()
                
                if len(y) > 10:  # Need enough data points
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x[:len(y)], y)
                    
                    trend_info[col] = {
                        "slope": slope,
                        "intercept": intercept,
                        "r_squared": r_value ** 2,
                        "p_value": p_value,
                        "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "no_trend",
                        "trend_strength": "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.4 else "weak"
                    }
                else:
                    trend_info[col] = {"trend_direction": "insufficient_data"}
                    
            except Exception as e:
                logger.warning(f"Failed to detect trend for {col}: {e}")
                trend_info[col] = {"error": str(e)}
        
        # Create trend features
        trend_features = {}
        
        for col, info in trend_info.items():
            if "slope" in info:
                # Create trend component
                x = np.arange(len(X[col]))
                trend_component = info["slope"] * x + info["intercept"]
                trend_features[f"{col}_trend"] = trend_component
                
                # Create detrended series
                detrended = X[col] - trend_component
                trend_features[f"{col}_detrended"] = detrended
        
        # Add trend features to dataframe
        for feature_name, feature_data in trend_features.items():
            X[feature_name] = feature_data
        
        metadata = {
            "trends_detected": trend_info,
            "trend_features_created": list(trend_features.keys())
        }
        
        self.trend_info = trend_info
        
        return X, metadata
    
    def _create_difference_features(self, X: pd.DataFrame, datetime_col: str,
                                  config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create difference features for stationarity"""
        logger.info("📉 Creating difference features")
        
        periods = config.get("difference_periods", [1, 7, 30])
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        diff_features = {}
        
        for col in numeric_cols:
            for period in periods:
                diff_feature_name = f"{col}_diff_{period}"
                diff_features[diff_feature_name] = X[col].diff(periods=period)
        
        # Add difference features to dataframe
        for feature_name, feature_data in diff_features.items():
            X[feature_name] = feature_data
        
        metadata = {
            "difference_features_created": list(diff_features.keys()),
            "periods_used": periods,
            "numeric_columns_used": list(numeric_cols)
        }
        
        return X, metadata
    
    def _handle_time_series_missing(self, X: pd.DataFrame, datetime_col: str,
                                  config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values in time series data"""
        logger.info("🔧 Handling time series missing values")
        
        strategy = config.get("missing_strategy", "interpolate")
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        missing_info = {}
        
        for col in numeric_cols:
            missing_count = X[col].isnull().sum()
            missing_ratio = missing_count / len(X)
            
            if missing_count > 0:
                missing_info[col] = {
                    "missing_count": missing_count,
                    "missing_ratio": missing_ratio,
                    "strategy_used": strategy
                }
                
                if strategy == "interpolate":
                    # Time series interpolation
                    X[col] = X[col].interpolate(method='time')
                elif strategy == "forward_fill":
                    X[col] = X[col].fillna(method='ffill')
                elif strategy == "backward_fill":
                    X[col] = X[col].fillna(method='bfill')
                elif strategy == "linear":
                    X[col] = X[col].interpolate(method='linear')
                else:
                    # Default to forward fill then backward fill
                    X[col] = X[col].fillna(method='ffill').fillna(method='bfill')
        
        metadata = {
            "missing_info": missing_info,
            "strategy": strategy,
            "total_missing_before": sum(X[numeric_cols].isnull().sum()),
            "total_missing_after": sum(X[numeric_cols].isnull().sum())
        }
        
        return X, metadata
    
    def _scale_time_series_features(self, X: pd.DataFrame, datetime_col: str,
                                  config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scale time series features"""
        logger.info("⚖️ Scaling time series features")
        
        method = config.get("scaling_method", "standard")
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            return X, {"method": "none"}
        
        # Fit and transform
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        self.scalers["global"] = scaler
        
        metadata = {
            "method": method,
            "features_scaled": list(numeric_cols),
            "scaler_fitted": True
        }
        
        return X, metadata
    
    def forecast_features(self, X: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Generate features for future periods"""
        logger.info(f"🔮 Generating features for {periods} future periods")
        
        # Get the last datetime
        datetime_col = self._detect_datetime_column(X)
        if datetime_col is None:
            raise ValueError("No datetime column found")
        
        last_datetime = X[datetime_col].max()
        
        # Create future datetime range
        freq = self._infer_frequency(X, datetime_col)
        future_datetimes = pd.date_range(
            start=last_datetime,
            periods=periods + 1,
            freq=freq
        )[1:]  # Exclude the last known datetime
        
        # Create future dataframe
        future_df = pd.DataFrame({datetime_col: future_datetimes})
        
        # Apply same feature engineering
        future_df, _ = self._extract_temporal_features(future_df, datetime_col, self._get_default_config())
        
        # For lag and rolling features, we need historical data
        # This is a simplified version - in practice, you'd need more sophisticated handling
        
        return future_df
    
    def _infer_frequency(self, X: pd.DataFrame, datetime_col: str) -> str:
        """Infer the frequency of the time series"""
        try:
            datetime_series = X[datetime_col].drop_duplicates().sort_values()
            
            if len(datetime_series) < 2:
                return 'D'  # Default to daily
            
            # Calculate time differences
            time_diffs = datetime_series.diff().dropna()
            
            # Most common time difference
            most_common_diff = time_diffs.mode().iloc[0]
            
            # Convert to pandas frequency string
            if most_common_diff.days >= 365:
                return 'Y'  # Yearly
            elif most_common_diff.days >= 30:
                return 'M'  # Monthly
            elif most_common_diff.days >= 7:
                return 'W'  # Weekly
            elif most_common_diff.days >= 1:
                return 'D'  # Daily
            elif most_common_diff.seconds >= 3600:
                return 'H'  # Hourly
            elif most_common_diff.seconds >= 60:
                return 'T'  # Minutely
            else:
                return 'S'  # Secondly
                
        except:
            return 'D'  # Default to daily
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of time series processing operations"""
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
                "features_created": latest.get("features_created", 0)
            },
            "seasonality_info": self.seasonality_info,
            "trend_info": self.trend_info,
            "all_steps_applied": list(set(
                step for session in self.processing_history 
                for step in session["processing_steps"]
            ))
        }
    
    def transform_new_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new time series data using fitted processors"""
        X_processed = X.copy()
        
        # Detect datetime column
        datetime_col = self._detect_datetime_column(X_processed)
        if datetime_col is None:
            raise ValueError("No datetime column found")
        
        X_processed[datetime_col] = pd.to_datetime(X_processed[datetime_col])
        X_processed = X_processed.sort_values(datetime_col)
        
        # Apply scaling if fitted
        if "global" in self.scalers:
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            X_processed[numeric_cols] = self.scalers["global"].transform(X_processed[numeric_cols])
        
        return X_processed
    
    def reset(self):
        """Reset all fitted processors"""
        self.scalers.clear()
        self.seasonality_info.clear()
        self.trend_info.clear()
        self.feature_importance.clear()
        self.processing_history.clear()
        logger.info("🔄 Time series processor reset")
