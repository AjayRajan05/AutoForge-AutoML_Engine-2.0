"""
🔧 Systemization Integration
Connects AutoForge with existing systemization modules
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

try:
    from .ab_testing import ABTestingFramework
    from .lightweight_monitoring import LightweightMonitor
    from .model_versioning import ModelVersioning
    ABTesting = ABTestingFramework
    LightweightMonitoring = LightweightMonitor
except ImportError:
    ABTestingFramework = None
    LightweightMonitor = None
    ABTesting = None
    LightweightMonitoring = None
    ModelVersioning = None

logger = logging.getLogger(__name__)


class SystemizationIntegrator:
    """
    Integration layer for systemization components
    """
    
    def __init__(self):
        """Initialize systemization integrator"""
        self.available_components = self._check_available_components()
        self.monitoring_data = []
        self.ab_test_results = []
        self.model_versions = []
        
    def _check_available_components(self) -> Dict[str, bool]:
        """Check which systemization components are available"""
        components = {
            'ab_testing': ABTestingFramework is not None,
            'monitoring': LightweightMonitor is not None,
            'model_versioning': ModelVersioning is not None
        }
        
        available_count = sum(components.values())
        logger.info(f"🔧 Available systemization components: {available_count}/{len(components)}")
        
        return components
    
    def setup_monitoring(self, autoforge_instance):
        """Setup monitoring for AutoForge"""
        try:
            if not self.available_components.get('monitoring', False):
                logger.warning("⚠️ Monitoring not available")
                return None
            
            logger.info("🔧 Setting up AutoForge monitoring...")
            
            monitor = LightweightMonitor(monitor_dir="monitoring")
            return monitor
            
        except Exception as e:
            logger.error(f"❌ Monitoring setup failed: {e}")
            return None
    
    def monitor_training(self, autoforge_instance, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Monitor AutoForge training process"""
        try:
            if not self.available_components.get('monitoring', False):
                return {'error': 'Monitoring not available'}
            
            # Start monitoring
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # Train AutoForge
            autoforge_instance.fit(X, y)
            
            # Collect metrics
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            metrics = {
                'training_time': end_time - start_time,
                'model_score': autoforge_instance.best_score,
                'feature_count': len(autoforge_instance.training_metadata.get('selected_features', [])),
                'memory_usage': end_memory - start_memory,
                'timestamp': time.time(),
                'dataset_size': X.shape,
                'model_type': type(autoforge_instance.best_model).__name__
            }
            
            # Store monitoring data
            self.monitoring_data.append(metrics)
            
            logger.info(f"🔧 Training monitored: {metrics['training_time']:.2f}s, score: {metrics['model_score']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Training monitoring failed: {e}")
            return {'error': str(e)}
    
    def setup_ab_testing(self, autoforge_instance, test_data: Dict[str, Any]) -> Optional[Any]:
        """Setup A/B testing for AutoForge"""
        try:
            if not self.available_components.get('ab_testing', False):
                logger.warning("⚠️ A/B testing not available")
                return None
            
            logger.info("🔧 Setting up AutoForge A/B testing...")
            
            # Create AB testing instance
            ab_tester = ABTestingFramework()
            ab_tester._autoforge_test_configs = {
                'control': {
                    'description': 'Current AutoForge configuration',
                    'config': autoforge_instance.config
                },
                'variant': {
                    'description': 'Optimized AutoForge configuration',
                    'config': self._get_optimized_config(autoforge_instance.config)
                }
            }
            
            return ab_tester
            
        except Exception as e:
            logger.error(f"❌ A/B testing setup failed: {e}")
            return None
    
    def run_ab_test(self, autoforge_instance, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Run A/B test on AutoForge configurations"""
        try:
            if not self.available_components.get('ab_testing', False):
                return {'error': 'A/B testing not available'}
            
            logger.info("🔧 Running AutoForge A/B test...")
            
            # Test control configuration
            control_config = autoforge_instance.config.copy()
            autoforge_control = type(autoforge_instance)(control_config)
            control_metrics = self.monitor_training(autoforge_control, X, y)
            
            # Test variant configuration
            variant_config = self._get_optimized_config(autoforge_instance.config)
            autoforge_variant = type(autoforge_instance)(variant_config)
            variant_metrics = self.monitor_training(autoforge_variant, X, y)
            
            # Calculate results
            results = {
                'control': control_metrics,
                'variant': variant_metrics,
                'winner': None,
                'improvement': 0
            }
            
            # Determine winner
            if control_metrics.get('model_score', 0) > variant_metrics.get('model_score', 0):
                results['winner'] = 'control'
                results['improvement'] = ((control_metrics['model_score'] - variant_metrics['model_score']) / variant_metrics['model_score']) * 100
            else:
                results['winner'] = 'variant'
                results['improvement'] = ((variant_metrics['model_score'] - control_metrics['model_score']) / control_metrics['model_score']) * 100
            
            # Store AB test results
            self.ab_test_results.append(results)
            
            logger.info(f"🔧 A/B test complete: {results['winner']} wins with {results['improvement']:.2f}% improvement")
            return results
            
        except Exception as e:
            logger.error(f"❌ A/B test failed: {e}")
            return {'error': str(e)}
    
    def setup_model_versioning(self, autoforge_instance) -> Optional[Any]:
        """Setup model versioning for AutoForge"""
        try:
            if not self.available_components.get('model_versioning', False):
                logger.warning("⚠️ Model versioning not available")
                return None
            
            logger.info("🔧 Setting up AutoForge model versioning...")
            
            # Create model versioning instance
            versioner = ModelVersioning()
            versioner._autoforge_metadata = {
                'model_type': 'AutoForge',
                'version': '1.0.0',
                'created_at': time.time(),
                'config': autoforge_instance.config,
                'training_metadata': autoforge_instance.training_metadata
            }
            
            return versioner
            
        except Exception as e:
            logger.error(f"❌ Model versioning setup failed: {e}")
            return None
    
    def save_model_version(self, autoforge_instance, version_name: str, 
                         description: str = "") -> Dict[str, Any]:
        """Save a version of the AutoForge model"""
        try:
            if not self.available_components.get('model_versioning', False):
                return {'error': 'Model versioning not available'}
            
            logger.info(f"🔧 Saving AutoForge model version: {version_name}")
            
            # Create version metadata
            version_info = {
                'version_name': version_name,
                'description': description,
                'timestamp': time.time(),
                'model_score': autoforge_instance.best_score,
                'model_type': type(autoforge_instance.best_model).__name__,
                'config': autoforge_instance.config,
                'training_metadata': autoforge_instance.training_metadata,
                'feature_importance': autoforge_instance.get_feature_importance()
            }
            
            # Save model (using existing model saver)
            model_path = autoforge_instance.save_model(version_name)
            version_info['model_path'] = model_path
            
            # Store version info
            self.model_versions.append(version_info)
            
            logger.info(f"🔧 Model version saved: {model_path}")
            return version_info
            
        except Exception as e:
            logger.error(f"❌ Model versioning failed: {e}")
            return {'error': str(e)}
    
    def get_systemization_summary(self) -> Dict[str, Any]:
        """Get summary of all systemization activities"""
        summary = {
            'available_components': self.available_components,
            'monitoring_sessions': len(self.monitoring_data),
            'ab_tests_run': len(self.ab_test_results),
            'model_versions_saved': len(self.model_versions),
            'latest_monitoring': self.monitoring_data[-1] if self.monitoring_data else None,
            'latest_ab_test': self.ab_test_results[-1] if self.ab_test_results else None,
            'latest_model_version': self.model_versions[-1] if self.model_versions else None
        }
        
        return summary
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
    
    def _get_optimized_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimized configuration for A/B testing"""
        optimized = base_config.copy()
        
        # Make some optimizations
        optimized['max_trials'] = optimized.get('max_trials', 50) * 1.5
        optimized['n_jobs'] = -1
        optimized['verbose'] = True
        
        return optimized
    
    def generate_systemization_report(self) -> str:
        """Generate systemization report"""
        lines = []
        
        # Header
        lines.append("🔧 AutoForge Systemization Report")
        lines.append("=" * 50)
        
        # Component availability
        lines.append("\n📋 Available Components:")
        for component, available in self.available_components.items():
            status = "✅" if available else "❌"
            lines.append(f"  {status} {component}")
        
        # Monitoring summary
        if self.monitoring_data:
            lines.append(f"\n📊 Monitoring Summary:")
            lines.append(f"  Sessions: {len(self.monitoring_data)}")
            latest = self.monitoring_data[-1]
            lines.append(f"  Latest Score: {latest.get('model_score', 'N/A'):.4f}")
            lines.append(f"  Latest Training Time: {latest.get('training_time', 'N/A'):.2f}s")
        
        # A/B testing summary
        if self.ab_test_results:
            lines.append(f"\n🧪 A/B Testing Summary:")
            lines.append(f"  Tests Run: {len(self.ab_test_results)}")
            latest = self.ab_test_results[-1]
            lines.append(f"  Latest Winner: {latest.get('winner', 'N/A')}")
            lines.append(f"  Latest Improvement: {latest.get('improvement', 0):.2f}%")
        
        # Model versioning summary
        if self.model_versions:
            lines.append(f"\n📦 Model Versioning Summary:")
            lines.append(f"  Versions Saved: {len(self.model_versions)}")
            latest = self.model_versions[-1]
            lines.append(f"  Latest Version: {latest.get('version_name', 'N/A')}")
            lines.append(f"  Latest Score: {latest.get('model_score', 'N/A'):.4f}")
        
        return "\n".join(lines)


# Global systemization integrator instance
systemization_integrator = SystemizationIntegrator()


def setup_autoforge_monitoring(autoforge_instance):
    """Setup monitoring for AutoForge"""
    return systemization_integrator.setup_monitoring(autoforge_instance)


def run_autoforge_ab_test(autoforge_instance, X: pd.DataFrame, y: pd.Series):
    """Run A/B test on AutoForge"""
    return systemization_integrator.run_ab_test(autoforge_instance, X, y)


def save_autoforge_version(autoforge_instance, version_name: str, description: str = ""):
    """Save AutoForge model version"""
    return systemization_integrator.save_model_version(autoforge_instance, version_name, description)
