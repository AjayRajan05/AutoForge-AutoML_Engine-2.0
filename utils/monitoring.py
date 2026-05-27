"""
📊 Monitoring System - Production monitoring for AutoForge
Tracks performance, errors, and system health
"""

import logging
import time
import json
import os
import platform
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading

from config.settings import MONITORING_CONFIG, get_config_value

logger = logging.getLogger(__name__)


class AutoForgeMonitor:
    """
    Production monitoring system for AutoForge
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize monitoring system
        
        Args:
            log_file: Optional custom log file path
        """
        self.log_file = log_file or get_config_value('monitoring', 'log_file', './logs/autoforge_monitoring.log')
        self.metrics = defaultdict(list)
        self.alerts = deque(maxlen=get_config_value('monitoring', 'max_alert_entries', 100))
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.alert_thresholds = get_config_value('monitoring', 'alert_thresholds', {
            'memory_usage': 80.0,
            'cpu_usage': 90.0,
            'disk_usage': 85.0,
            'execution_time': 300.0
        })
        self.max_log_entries = get_config_value('monitoring', 'max_log_entries', 1000)
        
        # Ensure log directory exists
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def _validate_inputs(self, execution_id: str, dataset_info: Dict[str, Any], 
                        config: Dict[str, Any], start_time: float, end_time: float) -> bool:
        """Validate inputs for log_execution method"""
        if not execution_id or not isinstance(execution_id, str):
            logger.error("Invalid execution_id: must be non-empty string")
            return False
        
        if not isinstance(dataset_info, dict):
            logger.error("Invalid dataset_info: must be dictionary")
            return False
        
        if not isinstance(config, dict):
            logger.error("Invalid config: must be dictionary")
            return False
        
        if not isinstance(start_time, (int, float)) or start_time <= 0:
            logger.error("Invalid start_time: must be positive number")
            return False
        
        if not isinstance(end_time, (int, float)) or end_time <= 0:
            logger.error("Invalid end_time: must be positive number")
            return False
        
        if end_time <= start_time:
            logger.error("Invalid time range: end_time must be greater than start_time")
            return False
        
        return True
    
    def _validate_performance_inputs(self, execution_id: str, model_name: str, 
                                 score: float, training_time: float) -> bool:
        """Validate inputs for log_model_performance method"""
        if not execution_id or not isinstance(execution_id, str):
            logger.error("Invalid execution_id: must be non-empty string")
            return False
        
        if not model_name or not isinstance(model_name, str):
            logger.error("Invalid model_name: must be non-empty string")
            return False
        
        if not isinstance(score, (int, float)):
            logger.error("Invalid score: must be numeric")
            return False
        
        if not isinstance(training_time, (int, float)) or training_time < 0:
            logger.error("Invalid training_time: must be non-negative number")
            return False
        
        return True
        
    def log_execution(self, execution_id: str, dataset_info: Dict[str, Any], 
                     config: Dict[str, Any], start_time: float, end_time: float,
                     success: bool, error_message: Optional[str] = None) -> bool:
        """Log execution metrics
        
        Args:
            execution_id: Unique identifier for execution
            dataset_info: Information about the dataset
            config: Configuration used for execution
            start_time: Start timestamp
            end_time: End timestamp
            success: Whether execution was successful
            error_message: Error message if failed
            
        Returns:
            True if logging was successful, False otherwise
        """
        if not self._validate_inputs(execution_id, dataset_info, config, start_time, end_time):
            return False
            
        try:
            execution_time = end_time - start_time
            
            log_entry = {
                'execution_id': execution_id,
                'timestamp': datetime.now().isoformat(),
                'dataset_info': {
                    'n_samples': dataset_info.get('n_samples', 0),
                    'n_features': dataset_info.get('n_features', 0),
                    'task_type': dataset_info.get('task_type', 'unknown')
                },
                'config': {
                    'max_trials': config.get('max_trials', 0),
                    'cv_folds': config.get('cv_folds', 0),
                    'models': config.get('models', [])
                },
                'performance': {
                    'execution_time': execution_time,
                    'success': success,
                    'error_message': error_message
                },
                'system_info': self._get_system_info()
            }
            
            with self.lock:
                self.metrics['executions'].append(log_entry)
                
                # Keep only configured number of executions in memory
                max_executions = self.max_log_entries['executions']
                if len(self.metrics['executions']) > max_executions:
                    self.metrics['executions'] = self.metrics['executions'][-max_executions:]
            
            # Write to file
            self._write_to_file()
            
            # Check for alerts
            self._check_alerts(log_entry)
            
            logger.info(f"Execution logged: {execution_id} - {execution_time:.2f}s - {'SUCCESS' if success else 'FAILED'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log execution: {e}")
            return False
    
    def log_model_performance(self, execution_id: str, model_name: str, 
                            score: float, training_time: float) -> bool:
        """Log model performance metrics
        
        Args:
            execution_id: Execution identifier
            model_name: Name of the model
            score: Performance score
            training_time: Time taken for training
            
        Returns:
            True if logging was successful, False otherwise
        """
        if not self._validate_performance_inputs(execution_id, model_name, score, training_time):
            return False
            
        try:
            performance_entry = {
                'execution_id': execution_id,
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'score': score,
                'training_time': training_time
            }
            
            with self.lock:
                self.metrics['model_performance'].append(performance_entry)
                
                # Keep only configured number of model performances
                max_performances = self.max_log_entries['model_performance']
                if len(self.metrics['model_performance']) > max_performances:
                    self.metrics['model_performance'] = self.metrics['model_performance'][-max_performances:]
            
            logger.info(f"Model performance logged: {model_name} - {score:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log model performance: {e}")
            return False
    
    def log_system_health(self) -> bool:
        """Log system health metrics
        
        Returns:
            True if logging was successful, False otherwise
        """
        try:
            import psutil
            
            health_entry = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'uptime_hours': (time.time() - self.start_time) / 3600
            }
            
            with self.lock:
                self.metrics['system_health'].append(health_entry)
                
                # Keep only configured number of health entries
                max_health = self.max_log_entries['system_health']
                if len(self.metrics['system_health']) > max_health:
                    self.metrics['system_health'] = self.metrics['system_health'][-max_health:]
            
            # Check for system alerts using configured thresholds
            cpu_threshold = self.alert_thresholds['cpu_percent']
            memory_threshold = self.alert_thresholds['memory_percent']
            disk_threshold = self.alert_thresholds['disk_percent']
            
            if health_entry['cpu_percent'] > cpu_threshold:
                self._create_alert('HIGH_CPU', f"CPU usage: {health_entry['cpu_percent']:.1f}%")
            
            if health_entry['memory_percent'] > memory_threshold:
                self._create_alert('HIGH_MEMORY', f"Memory usage: {health_entry['memory_percent']:.1f}%")
            
            if health_entry['disk_percent'] > disk_threshold:
                self._create_alert('HIGH_DISK', f"Disk usage: {health_entry['disk_percent']:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log system health: {e}")
            return False
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information
        
        Returns:
            Dictionary containing system information
        """
        try:
            
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'platform': platform.platform(),
                'python_version': platform.python_version()
            }
        except Exception as e:
            logger.warning(f"Failed to get system info: {e}")
            return {}
    
    def _check_alerts(self, execution_entry: Dict[str, Any]) -> bool:
        """Check for execution alerts
        
        Args:
            execution_entry: Execution data to check
            
        Returns:
            True if check completed successfully, False otherwise
        """
        try:
            execution_time = execution_entry['performance']['execution_time']
            success = execution_entry['performance']['success']
            
            # Alert for long execution times using configured threshold
            execution_threshold = self.alert_thresholds['execution_time_seconds']
            if execution_time > execution_threshold:
                self._create_alert('LONG_EXECUTION', f"Execution time: {execution_time:.1f}s")
            
            # Alert for failures
            if not success:
                self._create_alert('EXECUTION_FAILURE', execution_entry['performance'].get('error_message', 'Unknown error'))
            
            # Alert for too many trials without success using configured threshold
            max_trials_threshold = self.alert_thresholds['max_trials']
            if not success and execution_entry['config']['max_trials'] > max_trials_threshold:
                self._create_alert('EXCESSIVE_TRIALS', f"Failed after {execution_entry['config']['max_trials']} trials")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check alerts: {e}")
            return False
    
    def _create_alert(self, alert_type: str, message: str) -> bool:
        """Create an alert
        
        Args:
            alert_type: Type of alert
            message: Alert message
            
        Returns:
            True if alert was created successfully, False otherwise
        """
        if not alert_type or not message:
            logger.warning("Invalid alert parameters: alert_type and message must be non-empty")
            return False
            
        try:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': alert_type,
                'message': message,
                'severity': self._get_alert_severity(alert_type)
            }
            
            with self.lock:
                self.alerts.append(alert)
            
            logger.warning(f"ALERT [{alert_type}]: {message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            return False
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Get alert severity"""
        severity_map = {
            'HIGH_CPU': 'warning',
            'HIGH_MEMORY': 'warning',
            'HIGH_DISK': 'critical',
            'LONG_EXECUTION': 'warning',
            'EXECUTION_FAILURE': 'error',
            'EXCESSIVE_TRIALS': 'warning'
        }
        return severity_map.get(alert_type, 'info')
    
    def _write_to_file(self) -> bool:
        """Write metrics to file
        
        Returns:
            True if write was successful, False otherwise
        """
        if not self.log_file:
            logger.warning("No log file configured, skipping write")
            return False
            
        try:
            # Use context manager for proper file handling
            with open(self.log_file, 'w') as f:
                # Convert defaultdict to regular dict for JSON serialization
                metrics_dict = {k: list(v) for k, v in self.metrics.items()}
                metrics_dict['alerts'] = list(self.alerts)
                metrics_dict['last_updated'] = datetime.now().isoformat()
                
                json.dump(metrics_dict, f, indent=2, default=str)
                
            return True
                
        except (IOError, OSError) as e:
            logger.error(f"File I/O error while writing metrics: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to write metrics to file: {e}")
            return False
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        try:
            with self.lock:
                # Recent executions (last 24 hours)
                recent_time = datetime.now() - timedelta(hours=24)
                recent_executions = [
                    exec for exec in self.metrics.get('executions', [])
                    if datetime.fromisoformat(exec['timestamp']) > recent_time
                ]
                
                # Success rate
                total_executions = len(recent_executions)
                successful_executions = len([e for e in recent_executions if e['performance']['success']])
                success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
                
                # Average execution time
                execution_times = [e['performance']['execution_time'] for e in recent_executions]
                avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
                
                # Recent alerts
                recent_alerts = [
                    alert for alert in self.alerts
                    if datetime.fromisoformat(alert['timestamp']) > recent_time
                ]
                
                return {
                    'system_uptime_hours': (time.time() - self.start_time) / 3600,
                    'recent_executions_24h': total_executions,
                    'success_rate_24h': success_rate,
                    'avg_execution_time_24h': avg_execution_time,
                    'recent_alerts_24h': len(recent_alerts),
                    'critical_alerts': len([a for a in recent_alerts if a['severity'] == 'critical']),
                    'total_models_trained': len(self.metrics.get('model_performance', [])),
                    'latest_system_health': self.metrics.get('system_health', [])[-1] if self.metrics.get('system_health') else None
                }
                
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {"error": str(e)}
    
    def start_monitoring(self, interval_seconds: Optional[int] = None) -> bool:
        """Start continuous monitoring
        
        Args:
            interval_seconds: Custom interval in seconds
            
        Returns:
            True if monitoring started successfully, False otherwise
        """
        interval = interval_seconds or get_config_value('monitoring', 'health_check_interval', 60)
        
        if interval <= 0:
            logger.error(f"Invalid monitoring interval: {interval} seconds")
            return False
            
        def monitor_loop():
            while True:
                try:
                    self.log_system_health()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
                    time.sleep(interval)
        
        try:
            monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
            monitor_thread.start()
            logger.info(f"Monitoring started with {interval}s interval")
            return True
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False


# Global monitor instance
autoforge_monitor = AutoForgeMonitor()
