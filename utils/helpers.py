"""
🔧 AutoForge Utilities - Helper functions and utilities
"""

import logging
import pandas as pd
import numpy as np
import platform
import psutil
import time
import warnings
from typing import Dict, Any, List, Optional, Union
from functools import wraps

from config.settings import VALIDATION_CONFIG, get_config_value

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", format_string: Optional[str] = None):
    """Setup logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = get_config_value('logging', 'format_string')
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format=format_string,
        datefmt=get_config_value('logging', 'date_format')
    )


def suppress_warnings():
    """Suppress common warnings"""
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)


def validate_dataframe(df: pd.DataFrame, min_rows: Optional[int] = None, 
                       min_cols: Optional[int] = None) -> Dict[str, Any]:
    """Validate DataFrame structure and content
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required (uses config if None)
        min_cols: Minimum number of columns required (uses config if None)
        
    Returns:
        Dictionary containing validation results
        
    Raises:
        ValueError: If df is not a pandas DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    min_rows = min_rows or get_config_value('validation', 'min_dataframe_rows')
    min_cols = min_cols or get_config_value('validation', 'min_dataframe_cols')
    max_missing_ratio = get_config_value('validation', 'max_missing_ratio')
    max_duplicate_ratio = get_config_value('validation', 'max_duplicate_ratio')
    
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {
            'shape': df.shape,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
    }
    
    # Check basic structure
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['errors'].append("DataFrame is empty")
    
    if df.shape[0] < min_rows:
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"DataFrame has {df.shape[0]} rows, minimum {min_rows} required")
    
    if df.shape[1] < min_cols:
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"DataFrame has {df.shape[1]} columns, minimum {min_cols} required")
    
    # Check for missing values
    missing_ratio = df.isnull().sum().sum() / df.size
    if missing_ratio > max_missing_ratio:
        validation_result['warnings'].append(f"High missing value ratio: {missing_ratio:.2%}")
    
    # Check for duplicate rows
    duplicate_ratio = df.duplicated().sum() / len(df)
    if duplicate_ratio > max_duplicate_ratio:
        validation_result['warnings'].append(f"High duplicate ratio: {duplicate_ratio:.2%}")
    
    # Check memory usage
    max_memory_gb = get_config_value('validation', 'max_memory_usage_gb')
    memory_gb = df.memory_usage(deep=True).sum() / (1024**3)
    if memory_gb > max_memory_gb:
        validation_result['warnings'].append(f"High memory usage: {memory_gb:.2f}GB")
    
    return validation_result


def safe_convert_to_numeric(series: pd.Series, errors: str = 'coerce') -> pd.Series:
    """Safely convert series to numeric
    
    Args:
        series: Pandas series to convert
        errors: How to handle errors ('coerce', 'raise', 'ignore')
        
    Returns:
        Converted series
        
    Raises:
        ValueError: If series is not a pandas Series
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a pandas Series")
    
    if errors not in ['coerce', 'raise', 'ignore']:
        raise ValueError("errors must be one of 'coerce', 'raise', 'ignore'")
    
    return pd.to_numeric(series, errors=errors)


def get_memory_usage(obj: Any, deep: bool = True) -> Dict[str, float]:
    """Get memory usage information
    
    Args:
        obj: Object to measure memory usage
        deep: Whether to perform deep analysis
        
    Returns:
        Dictionary with memory usage in different units
    """
    if isinstance(obj, pd.DataFrame):
        memory_bytes = obj.memory_usage(deep=deep).sum()
    elif isinstance(obj, pd.Series):
        memory_bytes = obj.memory_usage(deep=deep)
    elif hasattr(obj, '__sizeof__'):
        memory_bytes = obj.__sizeof__()
    else:
        memory_bytes = 0
    
    return {
        'bytes': memory_bytes,
        'mb': memory_bytes / (1024 * 1024),
        'gb': memory_bytes / (1024 * 1024 * 1024)
    }


def timer_decorator(func):
    """Decorator to time function execution
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that logs execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
        return result
    
    return wrapper


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function on failure
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        Decorator function
        
    Raises:
        ValueError: If max_retries or delay are invalid
    """
    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    
    if not isinstance(delay, (int, float)) or delay < 0:
        raise ValueError("delay must be a non-negative number")
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")
            
            raise last_exception
        
        return wrapper
    return decorator


def create_sample_data(n_samples: int = None, n_features: int = None, 
                       task_type: str = 'classification', noise: float = None,
                       random_state: int = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create sample dataset for testing
    
    Args:
        n_samples: Number of samples (configurable)
        n_features: Number of features (configurable)
        task_type: Type of task ('classification' or 'regression')
        noise: Noise level (configurable)
        random_state: Random seed (configurable)
        
    Returns:
        Tuple of (X, y) data
    """
    # Use configurable defaults
    try:
        from config.settings import get_config_value
        n_samples = n_samples or get_config_value('sample_data', 'n_samples', 1000)
        n_features = n_features or get_config_value('sample_data', 'n_features', 10)
        noise = noise if noise is not None else get_config_value('sample_data', 'noise', 0.1)
        random_state = random_state if random_state is not None else get_config_value('sample_data', 'random_state', 42)
    except ImportError:
        # Fallback to hardcoded values
        n_samples = n_samples or 1000
        n_features = n_features or 10
        noise = noise if noise is not None else 0.1
        random_state = random_state if random_state is not None else 42
    
    np.random.seed(random_state)
    
    # Generate features
    data = {}
    for i in range(n_features):
        if i < n_features // 2:
            # Numeric features
            data[f'numeric_{i}'] = np.random.randn(n_samples)
        else:
            # Categorical features
            data[f'cat_{i}'] = np.random.choice(['A', 'B', 'C'], n_samples)
    
    X = pd.DataFrame(data)
    
    # Generate target
    if task_type == 'classification':
        # Create classification target
        scores = np.sum(X.select_dtypes(include=[np.number]), axis=1)
        scores += np.random.normal(0, noise, n_samples)
        y = pd.Series((scores > np.median(scores)).astype(int))
    else:
        # Create regression target
        y = pd.Series(np.sum(X.select_dtypes(include=[np.number]), axis=1) + 
                      np.random.normal(0, noise, n_samples))
    
    return X, y


def format_time(seconds: float) -> str:
    """Format time in human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}m {remaining_seconds:.0f}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(remaining_minutes)}m"


def format_number(number: Union[int, float], precision: int = 2) -> str:
    """Format number in human-readable format"""
    if isinstance(number, int):
        return f"{number:,}"
    else:
        return f"{number:,.{precision}f}"


def check_dependencies(required_packages: List[str]) -> Dict[str, bool]:
    """Check if required packages are available"""
    availability = {}
    
    for package in required_packages:
        try:
            __import__(package)
            availability[package] = True
        except ImportError:
            availability[package] = False
    
    return availability


def get_system_info() -> Dict[str, Any]:
    """Get system information
    
    Returns:
        Dictionary containing system information
    """
    try:
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_total_gb': psutil.disk_usage('/').total / (1024**3)
        }
    except Exception as e:
        logger.warning(f"Failed to get system info: {e}")
        return {}


class ProgressTracker:
    """Simple progress tracker"""
    
    def __init__(self, total_steps: int, description: str = "Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, step: int = 1, message: str = ""):
        """Update progress"""
        self.current_step += step
        progress = self.current_step / self.total_steps
        elapsed = time.time() - self.start_time
        
        if self.current_step <= self.total_steps:
            eta = elapsed / progress * (1 - progress) if progress > 0 else 0
            logger.info(f"📊 {self.description}: {self.current_step}/{self.total_steps} "
                       f"({progress:.1%}) - ETA: {format_time(eta)} {'- ' + message if message else ''}")
        else:
            logger.info(f"✅ {self.description}: Completed in {format_time(elapsed)}")
    
    def finish(self, message: str = "Completed"):
        """Mark as completed"""
        elapsed = time.time() - self.start_time
        logger.info(f"🎉 {self.description}: {message} in {format_time(elapsed)}")


def cache_result(cache_key: str, cache_dict: Dict[str, Any]):
    """Decorator to cache function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if cache_key in cache_dict:
                logger.debug(f"📋 Using cached result for {cache_key}")
                return cache_dict[cache_key]
            
            result = func(*args, **kwargs)
            cache_dict[cache_key] = result
            logger.debug(f"💾 Cached result for {cache_key}")
            return result
        
        return wrapper
    return decorator


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> Dict[str, Any]:
    """Validate configuration dictionary"""
    validation_result = {
        'is_valid': True,
        'missing_keys': [],
        'invalid_values': {}
    }
    
    # Check required keys
    for key in required_keys:
        if key not in config:
            validation_result['is_valid'] = False
            validation_result['missing_keys'].append(key)
    
    # Validate common configurations
    if 'max_time' in config and config['max_time'] <= 0:
        validation_result['invalid_values']['max_time'] = 'Must be positive'
    
    if 'max_trials' in config and config['max_trials'] <= 0:
        validation_result['invalid_values']['max_trials'] = 'Must be positive'
    
    if 'random_state' in config and not isinstance(config['random_state'], int):
        validation_result['invalid_values']['random_state'] = 'Must be integer'
    
    return validation_result
