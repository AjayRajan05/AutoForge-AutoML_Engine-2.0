"""
Configuration settings for AutoForge AutoML Engine
Centralized configuration management
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Cache settings
DEFAULT_CACHE_CONFIG = {
    "pipeline_cache_dir": str(CACHE_DIR / "pipeline"),
    "model_cache_dir": str(CACHE_DIR / "models"),
    "memory_limit_mb": 2048,
    "max_cache_size_gb": 10
}

# Monitoring settings
MONITORING_CONFIG = {
    "log_file": str(LOGS_DIR / "autoforge_monitoring.json"),
    "health_check_interval": 60,
    "max_log_entries": {
        "executions": 1000,
        "model_performance": 500,
        "system_health": 100,
        "alerts": 100
    },
    "alert_thresholds": {
        "cpu_percent": 90,
        "memory_percent": 90,
        "disk_percent": 90,
        "execution_time_seconds": 300,
        "max_trials": 50
    }
}

# Logging settings
LOGGING_CONFIG = {
    "level": "INFO",
    "format_string": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "log_file": str(LOGS_DIR / "autoforge.log")
}

# Performance settings
PERFORMANCE_CONFIG = {
    "memory_threshold": 0.8,  # 80%
    "max_execution_time": 3600,  # 1 hour
    "max_concurrent_jobs": 4,
    "cleanup_interval": 300  # 5 minutes
}

# Validation settings
VALIDATION_CONFIG = {
    "min_dataframe_rows": 1,
    "min_dataframe_cols": 1,
    "max_missing_ratio": 0.5,
    "max_duplicate_ratio": 0.3,
    "max_memory_usage_gb": 8
}


def get_config_value(section: str, key: str, default: Any = None) -> Any:
    """
    Get configuration value from environment variables or defaults
    
    Args:
        section: Configuration section (e.g., 'cache', 'monitoring')
        key: Configuration key
        default: Default value if not found
        
    Returns:
        Configuration value
    """
    env_key = f"AUTOFORGE_{section.upper()}_{key.upper()}"
    
    # Try environment variable first
    env_value = os.getenv(env_key)
    if env_value is not None:
        return env_value
    
    # Get from default configs
    configs = {
        "cache": DEFAULT_CACHE_CONFIG,
        "monitoring": MONITORING_CONFIG,
        "logging": LOGGING_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "validation": VALIDATION_CONFIG
    }
    
    return configs.get(section, {}).get(key, default)


def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        CACHE_DIR,
        LOGS_DIR,
        CONFIG_DIR,
        Path(DEFAULT_CACHE_CONFIG["pipeline_cache_dir"]),
        Path(DEFAULT_CACHE_CONFIG["model_cache_dir"])
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Initialize directories on import
ensure_directories()
