"""
Adaptive Resource Manager
Manages computational resources adaptively
"""

import logging
import psutil
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AdaptiveResourceManager:
    """
    Adaptive Resource Manager
    
    Automatically adjusts resource usage based on available system resources
    """
    
    def __init__(self):
        """Initialize adaptive resource manager"""
        self.resource_limits = self._detect_system_resources()
        self.current_usage = {}
        
    def _detect_system_resources(self) -> Dict[str, Any]:
        """Detect available system resources"""
        try:
            # Get system info
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()
            
            limits = {
                'cpu_cores': cpu_count,
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_usage_percent': memory.percent
            }
            
            # Set conservative limits
            limits['max_cpu_cores'] = max(1, cpu_count - 1)
            limits['max_memory_gb'] = limits['memory_available_gb'] * 0.8  # Use 80% of available memory
            
            logger.info(f"🖥️ Detected resources: {cpu_count} CPUs, {limits['memory_total_gb']:.1f}GB RAM")
            
            return limits
            
        except Exception as e:
            logger.error(f"❌ Failed to detect system resources: {e}")
            return {
                'cpu_cores': 4,
                'memory_total_gb': 8.0,
                'memory_available_gb': 4.0,
                'max_cpu_cores': 3,
                'max_memory_gb': 3.2
            }
    
    def adjust_parameters_for_resources(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust parameters based on available resources
        
        Args:
            parameters: Input parameters
            
        Returns:
            Resource-adjusted parameters
        """
        try:
            adjusted = parameters.copy()
            
            # Adjust n_estimators based on CPU cores
            if 'n_estimators' in adjusted:
                max_estimators = self.resource_limits['max_cpu_cores'] * 50
                adjusted['n_estimators'] = min(adjusted['n_estimators'], max_estimators)
            
            # Adjust cv_folds based on CPU cores
            if 'cv_folds' in adjusted:
                adjusted['cv_folds'] = min(adjusted['cv_folds'], self.resource_limits['max_cpu_cores'])
            
            # Adjust n_trials based on available memory
            if 'n_trials' in adjusted:
                max_trials = int(self.resource_limits['max_memory_gb'] * 10)
                adjusted['n_trials'] = min(adjusted['n_trials'], max_trials)
            
            logger.info(f"🔧 Adjusted parameters for available resources")
            
            return adjusted
            
        except Exception as e:
            logger.error(f"❌ Failed to adjust parameters: {e}")
            return parameters
    
    def get_search_budget(self, search_depth: str = "balanced", max_trials: Optional[int] = None,
                          max_time: Optional[float] = None) -> Dict[str, Any]:
        """Return resource-adjusted search budget for one model."""
        depth_trials = {"fast": 0, "balanced": 8, "deep": 50}
        base = max_trials if max_trials is not None else depth_trials.get(search_depth, 8)
        budget = {
            "max_trials_per_model": base,
            "timeout_per_model": max_time or (120 if search_depth == "deep" else 45),
        }
        adjusted = self.adjust_parameters_for_resources({"n_trials": base, "cv_folds": 5})
        budget["max_trials_per_model"] = adjusted.get("n_trials", base)
        budget["cv_folds"] = adjusted.get("cv_folds", 5)
        return budget

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            status = {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'resource_limits': self.resource_limits
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get resource status: {e}")
            return {'error': str(e)}

# Create global instance
adaptive_resource_manager = AdaptiveResourceManager()
