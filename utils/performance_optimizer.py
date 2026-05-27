"""
⚡ Performance Optimizer - Optimizes AutoForge for production use
Memory management, speed optimization, and resource efficiency
"""

import logging
import gc
import os
import threading
import time
from contextlib import contextmanager
from functools import wraps
from typing import Dict, Any, Optional, Callable
import psutil
import numpy as np
import pandas as pd

from config.settings import PERFORMANCE_CONFIG, get_config_value

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Optimizes AutoForge performance for production use
    """
    
    def __init__(self):
        """Initialize performance optimizer"""
        self.memory_threshold = get_config_value('performance', 'memory_threshold', 0.8)
        self.max_execution_time = get_config_value('performance', 'max_execution_time', 300)
        self.max_concurrent_jobs = get_config_value('performance', 'max_concurrent_jobs', 4)
        self.cleanup_interval = get_config_value('performance', 'cleanup_interval', 60)
        
        try:
            self.process = psutil.Process()
        except (ImportError, psutil.AccessDenied):
            self.process = None
            logger.warning("psutil not available or access denied, some features disabled")
        
        self.performance_history = []
        self.cleanup_thread = None
        self._start_cleanup_thread()
        
    def check_memory_usage(self) -> Dict[str, Any]:
        """
        Check current memory usage
        
        Returns:
            Dictionary with memory usage information
        """
        if not self.process:
            return {"error": "Process monitoring not available"}
        
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': memory_percent,
                'threshold_exceeded': memory_percent > self.memory_threshold * 100
            }
        except Exception as e:
            logger.error(f"Failed to check memory usage: {e}")
            return {}
    
    def cleanup_memory(self) -> bool:
        """
        Clean up memory usage
        
        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Clear performance history if too large
            max_history_size = get_config_value('performance', 'max_history_size', 1000)
            if len(self.performance_history) > max_history_size:
                keep_size = get_config_value('performance', 'keep_history_size', 500)
                self.performance_history = self.performance_history[-keep_size:]
            
            logger.info(f"Memory cleanup completed, collected {collected} objects")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup memory: {e}")
            return False
    
    def optimize_execution_speed(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize execution speed based on configuration
        
        Args:
            config: Current configuration
            
        Returns:
            Optimized configuration
        """
        try:
            optimized_config = config.copy()
            
            # Adjust parallel processing based on available resources
            if self.process:
                cpu_count = self.process.cpu_count()
                if 'n_jobs' in optimized_config and optimized_config['n_jobs'] == -1:
                    optimized_config['n_jobs'] = min(cpu_count, self.max_concurrent_jobs)
            
            # Enable memory optimization flags
            if 'memory_efficient' not in optimized_config:
                optimized_config['memory_efficient'] = True
            
            return optimized_config
            
        except Exception as e:
            logger.error(f"Failed to optimize execution speed: {e}")
            return config
    
    def monitor_performance(self, func_name: str, start_time: float, end_time: float):
        """
        Monitor and log performance metrics
        
        Args:
            func_name: Name of the function
            start_time: Start timestamp
            end_time: End timestamp
        """
        try:
            execution_time = end_time - start_time
            
            performance_entry = {
                'function': func_name,
                'execution_time': execution_time,
                'timestamp': end_time,
                'memory_usage': self.check_memory_usage()
            }
            
            self.performance_history.append(performance_entry)
            
            # Keep only recent performance history
            max_monitoring_history = get_config_value('performance', 'max_monitoring_history', 100)
            if len(self.performance_history) > max_monitoring_history:
                self.performance_history = self.performance_history[-max_monitoring_history:]
                
        except Exception as e:
            logger.error(f"Failed to monitor performance: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary
        
        Returns:
            Dictionary with performance summary
        """
        try:
            if not self.performance_history:
                return {"message": "No performance data available"}
            
            recent_history = self.performance_history[-10:]  # Last 10 executions
            
            avg_execution_time = sum(entry['execution_time'] for entry in recent_history) / len(recent_history)
            max_execution_time = max(entry['execution_time'] for entry in recent_history)
            
            return {
                'total_executions': len(self.performance_history),
                'avg_execution_time': avg_execution_time,
                'max_execution_time': max_execution_time,
                'recent_performance': recent_history
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self.cleanup_memory()
                except Exception as e:
                    logger.error(f"Cleanup thread error: {e}")
        
        self.cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        logger.info("Performance optimizer cleanup thread started")


def performance_monitor(func: Callable) -> Callable:
    """
    Decorator to monitor function performance
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with performance monitoring
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        optimizer = PerformanceOptimizer()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            optimizer.monitor_performance(func.__name__, start_time, end_time)
            return result
            
        except Exception as e:
            end_time = time.time()
            optimizer.monitor_performance(f"{func.__name__}_failed", start_time, end_time)
            raise
    
    return wrapper


@contextmanager
def memory_limit_context(limit_mb: Optional[int] = None):
    """
    Context manager to limit memory usage
    
    Args:
        limit_mb: Memory limit in MB (uses config if None)
    """
    limit = limit_mb or get_config_value('performance', 'max_memory_usage_gb') * 1024
    
    try:
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)
        
        yield
        
        current_memory = process.memory_info().rss / (1024 * 1024)
        if current_memory - initial_memory > limit:
            logger.warning(f"Memory limit exceeded: {current_memory - initial_memory:.1f}MB > {limit}MB")
            gc.collect()
            
    except Exception as e:
        logger.error(f"Memory limit context error: {e}")


def optimize_dataframe_memory(df) -> tuple:
    """
    Optimize DataFrame memory usage
    
    Args:
        df: pandas DataFrame to optimize
        
    Returns:
        Tuple of (optimized_df, memory_saved_mb)
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        start_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        optimized_df = df.copy()
        
        # Optimize numeric columns
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    optimized_df[col] = optimized_df[col].astype('uint8')
                elif col_max < 65535:
                    optimized_df[col] = optimized_df[col].astype('uint16')
                elif col_max < 4294967295:
                    optimized_df[col] = optimized_df[col].astype('uint32')
            else:
                if col_min > -128 and col_max < 127:
                    optimized_df[col] = optimized_df[col].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    optimized_df[col] = optimized_df[col].astype('int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    optimized_df[col] = optimized_df[col].astype('int32')
        
        # Optimize float columns
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # Optimize object columns
        for col in optimized_df.select_dtypes(include=['object']).columns:
            num_unique_values = len(optimized_df[col].unique())
            num_total_values = len(optimized_df[col])
            
            if num_unique_values / num_total_values < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
        
        end_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
        memory_saved = start_memory - end_memory
        
        logger.info(f"DataFrame memory optimized: {memory_saved:.2f}MB saved ({start_memory:.2f}MB -> {end_memory:.2f}MB)")
        
        return optimized_df, memory_saved
        
    except Exception as e:
        logger.error(f"Failed to optimize DataFrame memory: {e}")
        return df, 0.0


# Global performance optimizer
performance_optimizer = PerformanceOptimizer()
