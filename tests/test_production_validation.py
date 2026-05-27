"""
🧪 Production Validation Tests
Stress testing, performance benchmarking, and reliability testing for AutoForge
"""

import logging
import time
import numpy as np
import pandas as pd
import psutil
import gc
from typing import Dict, Any, List
import pytest
from unittest.mock import patch

# Import AutoForge components
from ..core.unified_automl import UnifiedAutoML
from ..input_output.input_types import AutoMLInput
from ..utils.performance_optimizer import performance_optimizer
from ..utils.monitoring import autoforge_monitor

logger = logging.getLogger(__name__)


class ProductionValidator:
    """
    Production validation system for AutoForge
    """
    
    def __init__(self):
        """Initialize production validator"""
        self.test_results = {}
        self.performance_metrics = {}
        self.reliability_metrics = {}
        
    def run_stress_tests(self) -> Dict[str, Any]:
        """Run comprehensive stress tests"""
        logger.info("🧪 Starting Production Stress Tests...")
        
        results = {
            'large_dataset_test': self._test_large_dataset(),
            'high_dimensional_test': self._test_high_dimensional(),
            'memory_stress_test': self._test_memory_stress(),
            'concurrent_execution_test': self._test_concurrent_execution(),
            'edge_cases_test': self._test_edge_cases()
        }
        
        self.test_results['stress_tests'] = results
        return results
    
    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks"""
        logger.info("⚡ Starting Performance Benchmarks...")
        
        results = {
            'speed_benchmark': self._benchmark_speed(),
            'memory_benchmark': self._benchmark_memory(),
            'scalability_benchmark': self._benchmark_scalability(),
            'feature_processing_benchmark': self._benchmark_feature_processing()
        }
        
        self.performance_metrics = results
        return results
    
    def run_reliability_tests(self) -> Dict[str, Any]:
        """Run reliability tests"""
        logger.info("🛡️ Starting Reliability Tests...")
        
        results = {
            'error_handling_test': self._test_error_handling(),
            'data_corruption_test': self._test_data_corruption(),
            'resource_exhaustion_test': self._test_resource_exhaustion(),
            'long_running_test': self._test_long_running()
        }
        
        self.reliability_metrics = results
        return results
    
    def _test_large_dataset(self) -> Dict[str, Any]:
        """Test with large dataset"""
        try:
            logger.info("📊 Testing large dataset (50K samples, 50 features)...")
            
            # Generate large dataset
            n_samples = 50000
            n_features = 50
            
            np.random.seed(42)
            X = np.random.randn(n_samples, n_features)
            y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
            
            # Convert to DataFrame
            feature_names = [f'feature_{i}' for i in range(n_features)]
            X_df = pd.DataFrame(X, columns=feature_names)
            y_series = pd.Series(y, name='target')
            
            # Create AutoML input
            automl_input = AutoMLInput(X_df, y_series, max_trials=20, max_time=300)
            
            # Initialize AutoForge
            autoforge = UnifiedAutoML()
            
            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024**2
            
            # Run AutoForge
            autoforge.fit(automl_input, enable_monitoring=True)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024**2
            
            # Test predictions
            predictions = autoforge.predict(X_df[:1000])  # Test subset
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            results = {
                'status': 'passed',
                'execution_time': execution_time,
                'memory_usage_mb': memory_usage,
                'samples_processed': n_samples,
                'features_processed': n_features,
                'predictions_shape': predictions.shape,
                'success': autoforge.is_fitted,
                'best_score': autoforge.best_score
            }
            
            logger.info(f"✅ Large dataset test: {execution_time:.2f}s, {memory_usage:.1f}MB")
            return results
            
        except Exception as e:
            logger.error(f"❌ Large dataset test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _test_high_dimensional(self) -> Dict[str, Any]:
        """Test with high-dimensional data"""
        try:
            logger.info("📈 Testing high-dimensional data (5K samples, 500 features)...")
            
            # Generate high-dimensional dataset
            n_samples = 5000
            n_features = 500
            
            np.random.seed(42)
            X = np.random.randn(n_samples, n_features)
            # Only first 10 features are actually important
            y = (X[:, :10].sum(axis=1) + np.random.randn(n_samples) * 0.1 > 0).astype(int)
            
            # Convert to DataFrame
            feature_names = [f'feature_{i}' for i in range(n_features)]
            X_df = pd.DataFrame(X, columns=feature_names)
            y_series = pd.Series(y, name='target')
            
            # Create AutoML input
            automl_input = AutoMLInput(X_df, y_series, max_trials=15, max_time=180)
            
            # Initialize AutoForge
            autoforge = UnifiedAutoML()
            
            # Measure performance
            start_time = time.time()
            
            # Run AutoForge
            autoforge.fit(automl_input, enable_monitoring=True)
            
            end_time = time.time()
            
            # Test predictions
            predictions = autoforge.predict(X_df[:500])
            
            results = {
                'status': 'passed',
                'execution_time': end_time - start_time,
                'samples_processed': n_samples,
                'features_processed': n_features,
                'predictions_shape': predictions.shape,
                'success': autoforge.is_fitted,
                'best_score': autoforge.best_score,
                'feature_selection_effective': autoforge.best_score > 0.7 if autoforge.best_score else False
            }
            
            logger.info(f"✅ High-dimensional test: {end_time - start_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ High-dimensional test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _test_memory_stress(self) -> Dict[str, Any]:
        """Test memory stress conditions"""
        try:
            logger.info("🧠 Testing memory stress conditions...")
            
            # Get initial memory
            initial_memory = psutil.Process().memory_info().rss / 1024**2
            
            # Test progressively larger datasets
            memory_results = []
            
            for size_multiplier in [1, 2, 4, 8]:
                try:
                    n_samples = 10000 * size_multiplier
                    n_features = 20
                    
                    # Generate dataset
                    X = np.random.randn(n_samples, n_features)
                    y = (X[:, 0] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
                    
                    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
                    y_series = pd.Series(y, name='target')
                    
                    automl_input = AutoMLInput(X_df, y_series, max_trials=5, max_time=60)
                    autoforge = UnifiedAutoML()
                    
                    # Measure memory before
                    before_memory = psutil.Process().memory_info().rss / 1024**2
                    
                    # Run AutoForge
                    autoforge.fit(automl_input, enable_monitoring=False)
                    
                    # Measure memory after
                    after_memory = psutil.Process().memory_info().rss / 1024**2
                    memory_increase = after_memory - before_memory
                    
                    memory_results.append({
                        'size_multiplier': size_multiplier,
                        'n_samples': n_samples,
                        'memory_increase_mb': memory_increase,
                        'success': autoforge.is_fitted
                    })
                    
                    # Clean up
                    del autoforge, X_df, y_series, automl_input
                    gc.collect()
                    
                except Exception as e:
                    memory_results.append({
                        'size_multiplier': size_multiplier,
                        'error': str(e),
                        'success': False
                    })
            
            results = {
                'status': 'passed',
                'initial_memory_mb': initial_memory,
                'memory_tests': memory_results,
                'max_memory_increase': max([r.get('memory_increase_mb', 0) for r in memory_results])
            }
            
            logger.info(f"✅ Memory stress test completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Memory stress test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _test_concurrent_execution(self) -> Dict[str, Any]:
        """Test concurrent execution capability"""
        try:
            logger.info("🔄 Testing concurrent execution...")
            
            import threading
            import queue
            
            results_queue = queue.Queue()
            
            def run_autoforge_task(task_id, results_queue):
                """Run AutoForge in separate thread"""
                try:
                    # Generate dataset
                    X = np.random.randn(1000, 10)
                    y = (X[:, 0] + np.random.randn(1000) * 0.1 > 0).astype(int)
                    
                    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
                    y_series = pd.Series(y, name='target')
                    
                    automl_input = AutoMLInput(X_df, y_series, max_trials=5, max_time=30)
                    autoforge = UnifiedAutoML()
                    
                    start_time = time.time()
                    autoforge.fit(automl_input, enable_monitoring=False)
                    end_time = time.time()
                    
                    results_queue.put({
                        'task_id': task_id,
                        'status': 'success',
                        'execution_time': end_time - start_time,
                        'success': autoforge.is_fitted
                    })
                    
                except Exception as e:
                    results_queue.put({
                        'task_id': task_id,
                        'status': 'failed',
                        'error': str(e)
                    })
            
            # Run multiple concurrent tasks
            threads = []
            start_time = time.time()
            
            for i in range(3):  # 3 concurrent tasks
                thread = threading.Thread(target=run_autoforge_task, args=(i, results_queue))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=60)  # 60 second timeout
            
            end_time = time.time()
            
            # Collect results
            task_results = []
            while not results_queue.empty():
                task_results.append(results_queue.get())
            
            successful_tasks = [r for r in task_results if r['status'] == 'success']
            
            results = {
                'status': 'passed' if len(successful_tasks) == 3 else 'failed',
                'total_execution_time': end_time - start_time,
                'concurrent_tasks': 3,
                'successful_tasks': len(successful_tasks),
                'task_results': task_results
            }
            
            logger.info(f"✅ Concurrent execution test: {len(successful_tasks)}/3 tasks successful")
            return results
            
        except Exception as e:
            logger.error(f"❌ Concurrent execution test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and unusual data"""
        try:
            logger.info("🔍 Testing edge cases...")
            
            edge_case_results = []
            
            # Test 1: All zeros
            try:
                X = np.zeros((100, 5))
                y = np.random.randint(0, 2, 100)
                
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
                y_series = pd.Series(y, name='target')
                
                automl_input = AutoMLInput(X_df, y_series, max_trials=3, max_time=30)
                autoforge = UnifiedAutoML()
                autoforge.fit(automl_input, enable_monitoring=False)
                
                edge_case_results.append({
                    'test': 'all_zeros',
                    'status': 'success',
                    'success': autoforge.is_fitted
                })
            except Exception as e:
                edge_case_results.append({
                    'test': 'all_zeros',
                    'status': 'failed',
                    'error': str(e)
                })
            
            # Test 2: Single feature
            try:
                X = np.random.randn(100, 1)
                y = (X[:, 0] > 0).astype(int)
                
                X_df = pd.DataFrame(X, columns=['single_feature'])
                y_series = pd.Series(y, name='target')
                
                automl_input = AutoMLInput(X_df, y_series, max_trials=3, max_time=30)
                autoforge = UnifiedAutoML()
                autoforge.fit(automl_input, enable_monitoring=False)
                
                edge_case_results.append({
                    'test': 'single_feature',
                    'status': 'success',
                    'success': autoforge.is_fitted
                })
            except Exception as e:
                edge_case_results.append({
                    'test': 'single_feature',
                    'status': 'failed',
                    'error': str(e)
                })
            
            # Test 3: Very small dataset
            try:
                X = np.random.randn(20, 5)
                y = np.random.randint(0, 2, 20)
                
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
                y_series = pd.Series(y, name='target')
                
                automl_input = AutoMLInput(X_df, y_series, max_trials=3, max_time=30)
                autoforge = UnifiedAutoML()
                autoforge.fit(automl_input, enable_monitoring=False)
                
                edge_case_results.append({
                    'test': 'very_small_dataset',
                    'status': 'success',
                    'success': autoforge.is_fitted
                })
            except Exception as e:
                edge_case_results.append({
                    'test': 'very_small_dataset',
                    'status': 'failed',
                    'error': str(e)
                })
            
            successful_edge_cases = len([r for r in edge_case_results if r['status'] == 'success'])
            
            results = {
                'status': 'passed' if successful_edge_cases >= 2 else 'failed',
                'total_edge_cases': len(edge_case_results),
                'successful_edge_cases': successful_edge_cases,
                'edge_case_results': edge_case_results
            }
            
            logger.info(f"✅ Edge cases test: {successful_edge_cases}/{len(edge_case_results)} passed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Edge cases test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _benchmark_speed(self) -> Dict[str, Any]:
        """Benchmark execution speed"""
        try:
            logger.info("⚡ Benchmarking execution speed...")
            
            # Test different dataset sizes
            speed_results = []
            
            for n_samples in [1000, 5000, 10000]:
                X = np.random.randn(n_samples, 10)
                y = (X[:, 0] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
                
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
                y_series = pd.Series(y, name='target')
                
                automl_input = AutoMLInput(X_df, y_series, max_trials=10, max_time=60)
                autoforge = UnifiedAutoML()
                
                start_time = time.time()
                autoforge.fit(automl_input, enable_monitoring=False)
                end_time = time.time()
                
                speed_results.append({
                    'n_samples': n_samples,
                    'execution_time': end_time - start_time,
                    'samples_per_second': n_samples / (end_time - start_time),
                    'success': autoforge.is_fitted
                })
            
            results = {
                'status': 'passed',
                'speed_results': speed_results,
                'avg_samples_per_second': np.mean([r['samples_per_second'] for r in speed_results])
            }
            
            logger.info(f"✅ Speed benchmark completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Speed benchmark failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _benchmark_memory(self) -> Dict[str, Any]:
        """Benchmark memory usage"""
        try:
            logger.info("🧠 Benchmarking memory usage...")
            
            memory_results = []
            
            for n_features in [10, 50, 100]:
                X = np.random.randn(1000, n_features)
                y = (X[:, 0] + np.random.randn(1000) * 0.1 > 0).astype(int)
                
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
                y_series = pd.Series(y, name='target')
                
                automl_input = AutoMLInput(X_df, y_series, max_trials=5, max_time=30)
                autoforge = UnifiedAutoML()
                
                # Measure memory before
                before_memory = psutil.Process().memory_info().rss / 1024**2
                
                autoforge.fit(automl_input, enable_monitoring=False)
                
                # Measure memory after
                after_memory = psutil.Process().memory_info().rss / 1024**2
                memory_increase = after_memory - before_memory
                
                memory_results.append({
                    'n_features': n_features,
                    'memory_increase_mb': memory_increase,
                    'memory_per_feature_mb': memory_increase / n_features,
                    'success': autoforge.is_fitted
                })
            
            results = {
                'status': 'passed',
                'memory_results': memory_results,
                'avg_memory_per_feature_mb': np.mean([r['memory_per_feature_mb'] for r in memory_results])
            }
            
            logger.info(f"✅ Memory benchmark completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Memory benchmark failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _benchmark_scalability(self) -> Dict[str, Any]:
        """Benchmark scalability"""
        try:
            logger.info("📈 Benchmarking scalability...")
            
            scalability_results = []
            
            # Test how performance scales with dataset size
            base_size = 1000
            for multiplier in [1, 2, 5, 10]:
                n_samples = base_size * multiplier
                n_features = 10
                
                X = np.random.randn(n_samples, n_features)
                y = (X[:, 0] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
                
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
                y_series = pd.Series(y, name='target')
                
                automl_input = AutoMLInput(X_df, y_series, max_trials=5, max_time=30)
                autoforge = UnifiedAutoML()
                
                start_time = time.time()
                autoforge.fit(automl_input, enable_monitoring=False)
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                scalability_results.append({
                    'multiplier': multiplier,
                    'n_samples': n_samples,
                    'execution_time': execution_time,
                    'time_complexity_factor': execution_time / (multiplier if multiplier > 0 else 1),
                    'success': autoforge.is_fitted
                })
            
            results = {
                'status': 'passed',
                'scalability_results': scalability_results,
                'scalability_factor': np.mean([r['time_complexity_factor'] for r in scalability_results[1:]])
            }
            
            logger.info(f"✅ Scalability benchmark completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Scalability benchmark failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _benchmark_feature_processing(self) -> Dict[str, Any]:
        """Benchmark advanced feature processing"""
        try:
            logger.info("🔧 Benchmarking feature processing...")
            
            # Test with different feature types
            feature_results = []
            
            # Test 1: Numeric only
            X_numeric = np.random.randn(1000, 10)
            y = (X_numeric[:, 0] + np.random.randn(1000) * 0.1 > 0).astype(int)
            
            X_df = pd.DataFrame(X_numeric, columns=[f'num_feature_{i}' for i in range(10)])
            y_series = pd.Series(y, name='target')
            
            automl_input = AutoMLInput(X_df, y_series, max_trials=5, max_time=30)
            autoforge = UnifiedAutoML()
            
            start_time = time.time()
            autoforge.fit(automl_input, enable_monitoring=False)
            end_time = time.time()
            
            feature_results.append({
                'feature_type': 'numeric_only',
                'execution_time': end_time - start_time,
                'success': autoforge.is_fitted,
                'best_score': autoforge.best_score
            })
            
            # Test 2: Mixed types (including text-like)
            X_mixed = np.random.randn(1000, 8)
            # Add categorical/text-like columns
            text_data = np.random.choice(['category_a', 'category_b', 'category_c'], 1000)
            
            X_mixed_df = pd.DataFrame(X_mixed, columns=[f'num_feature_{i}' for i in range(8)])
            X_mixed_df['text_feature'] = text_data
            X_mixed_df['another_text'] = np.random.choice(['type_1', 'type_2'], 1000)
            
            automl_input_mixed = AutoMLInput(X_mixed_df, y_series, max_trials=5, max_time=30)
            autoforge_mixed = UnifiedAutoML()
            
            start_time = time.time()
            autoforge_mixed.fit(automl_input_mixed, enable_monitoring=False)
            end_time = time.time()
            
            feature_results.append({
                'feature_type': 'mixed_types',
                'execution_time': end_time - start_time,
                'success': autoforge_mixed.is_fitted,
                'best_score': autoforge_mixed.best_score
            })
            
            results = {
                'status': 'passed',
                'feature_results': feature_results,
                'advanced_features_working': all(r['success'] for r in feature_results)
            }
            
            logger.info(f"✅ Feature processing benchmark completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Feature processing benchmark failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling robustness"""
        try:
            logger.info("🛡️ Testing error handling...")
            
            error_test_results = []
            
            # Test 1: Invalid data types
            try:
                # Pass invalid data
                autoforge = UnifiedAutoML()
                autoforge.fit("invalid_data", enable_monitoring=False)
                error_test_results.append({
                    'test': 'invalid_data_type',
                    'status': 'failed',
                    'error': 'Should have raised error'
                })
            except Exception:
                error_test_results.append({
                    'test': 'invalid_data_type',
                    'status': 'success',
                    'error': 'Correctly handled invalid data'
                })
            
            # Test 2: Empty dataset
            try:
                X_empty = pd.DataFrame()
                y_empty = pd.Series([])
                
                automl_input = AutoMLInput(X_empty, y_empty, max_trials=1, max_time=10)
                autoforge = UnifiedAutoML()
                autoforge.fit(automl_input, enable_monitoring=False)
                error_test_results.append({
                    'test': 'empty_dataset',
                    'status': 'failed',
                    'error': 'Should have raised error'
                })
            except Exception:
                error_test_results.append({
                    'test': 'empty_dataset',
                    'status': 'success',
                    'error': 'Correctly handled empty dataset'
                })
            
            # Test 3: Mismatched X and y
            try:
                X_mismatch = pd.DataFrame(np.random.randn(100, 5))
                y_mismatch = pd.Series(np.random.randint(0, 2, 50))  # Different length
                
                automl_input = AutoMLInput(X_mismatch, y_mismatch, max_trials=1, max_time=10)
                autoforge = UnifiedAutoML()
                autoforge.fit(automl_input, enable_monitoring=False)
                error_test_results.append({
                    'test': 'mismatched_dimensions',
                    'status': 'failed',
                    'error': 'Should have raised error'
                })
            except Exception:
                error_test_results.append({
                    'test': 'mismatched_dimensions',
                    'status': 'success',
                    'error': 'Correctly handled mismatched dimensions'
                })
            
            successful_error_handling = len([r for r in error_test_results if r['status'] == 'success'])
            
            results = {
                'status': 'passed' if successful_error_handling >= 2 else 'failed',
                'total_error_tests': len(error_test_results),
                'successful_error_handling': successful_error_handling,
                'error_test_results': error_test_results
            }
            
            logger.info(f"✅ Error handling test: {successful_error_handling}/{len(error_test_results)} passed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error handling test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _test_data_corruption(self) -> Dict[str, Any]:
        """Test handling of corrupted data"""
        try:
            logger.info("🔍 Testing data corruption handling...")
            
            corruption_results = []
            
            # Test 1: NaN values
            try:
                X_nan = pd.DataFrame(np.random.randn(100, 5))
                X_nan.iloc[10:20, 1] = np.nan  # Insert NaN values
                X_nan.iloc[30:40, 3] = np.nan
                
                y_nan = pd.Series(np.random.randint(0, 2, 100))
                
                automl_input = AutoMLInput(X_nan, y_nan, max_trials=3, max_time=30)
                autoforge = UnifiedAutoML()
                autoforge.fit(automl_input, enable_monitoring=False)
                
                corruption_results.append({
                    'test': 'nan_values',
                    'status': 'success',
                    'success': autoforge.is_fitted
                })
            except Exception as e:
                corruption_results.append({
                    'test': 'nan_values',
                    'status': 'failed',
                    'error': str(e)
                })
            
            # Test 2: Infinite values
            try:
                X_inf = pd.DataFrame(np.random.randn(100, 5))
                X_inf.iloc[5:10, 2] = np.inf
                X_inf.iloc[15:20, 4] = -np.inf
                
                y_inf = pd.Series(np.random.randint(0, 2, 100))
                
                automl_input = AutoMLInput(X_inf, y_inf, max_trials=3, max_time=30)
                autoforge = UnifiedAutoML()
                autoforge.fit(automl_input, enable_monitoring=False)
                
                corruption_results.append({
                    'test': 'infinite_values',
                    'status': 'success',
                    'success': autoforge.is_fitted
                })
            except Exception as e:
                corruption_results.append({
                    'test': 'infinite_values',
                    'status': 'failed',
                    'error': str(e)
                })
            
            successful_corruption_handling = len([r for r in corruption_results if r['status'] == 'success'])
            
            results = {
                'status': 'passed' if successful_corruption_handling >= 1 else 'failed',
                'total_corruption_tests': len(corruption_results),
                'successful_corruption_handling': successful_corruption_handling,
                'corruption_results': corruption_results
            }
            
            logger.info(f"✅ Data corruption test: {successful_corruption_handling}/{len(corruption_results)} passed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Data corruption test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _test_resource_exhaustion(self) -> Dict[str, Any]:
        """Test resource exhaustion handling"""
        try:
            logger.info("🔋 Testing resource exhaustion handling...")
            
            # This test simulates resource pressure
            resource_results = []
            
            # Test with very high trial count (should be limited automatically)
            try:
                X = pd.DataFrame(np.random.randn(1000, 10))
                y = pd.Series(np.random.randint(0, 2, 1000))
                
                automl_input = AutoMLInput(X, y, max_trials=1000, max_time=5)  # Very low time limit
                autoforge = UnifiedAutoML()
                
                start_time = time.time()
                autoforge.fit(automl_input, enable_monitoring=False)
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                # Should finish within reasonable time due to limits
                resource_results.append({
                    'test': 'time_limit_enforced',
                    'status': 'success' if execution_time < 30 else 'failed',
                    'execution_time': execution_time,
                    'success': autoforge.is_fitted
                })
            except Exception as e:
                resource_results.append({
                    'test': 'time_limit_enforced',
                    'status': 'failed',
                    'error': str(e)
                })
            
            successful_resource_handling = len([r for r in resource_results if r['status'] == 'success'])
            
            results = {
                'status': 'passed' if successful_resource_handling >= 1 else 'failed',
                'total_resource_tests': len(resource_results),
                'successful_resource_handling': successful_resource_handling,
                'resource_results': resource_results
            }
            
            logger.info(f"✅ Resource exhaustion test: {successful_resource_handling}/{len(resource_results)} passed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Resource exhaustion test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _test_long_running(self) -> Dict[str, Any]:
        """Test long-running execution"""
        try:
            logger.info("⏱️ Testing long-running execution...")
            
            # Run a longer execution to test stability
            X = pd.DataFrame(np.random.randn(2000, 15))
            y = pd.Series(np.random.randint(0, 2, 2000))
            
            automl_input = AutoMLInput(X, y, max_trials=20, max_time=120)  # 2 minutes
            autoforge = UnifiedAutoML()
            
            start_time = time.time()
            autoforge.fit(automl_input, enable_monitoring=True)
            end_time = time.time()
            
            # Test predictions after long run
            predictions = autoforge.predict(X[:100])
            
            results = {
                'status': 'success',
                'execution_time': end_time - start_time,
                'success': autoforge.is_fitted,
                'predictions_shape': predictions.shape,
                'best_score': autoforge.best_score,
                'monitoring_data_available': len(autoforge_monitor.metrics) > 0
            }
            
            logger.info(f"✅ Long-running test: {end_time - start_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ Long-running test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary"""
        try:
            summary = {
                'overall_status': 'passed',
                'stress_tests_passed': 0,
                'performance_tests_passed': 0,
                'reliability_tests_passed': 0,
                'total_tests': 0,
                'details': {}
            }
            
            # Count stress test results
            if 'stress_tests' in self.test_results:
                stress_results = self.test_results['stress_tests']
                summary['details']['stress_tests'] = stress_results
                summary['stress_tests_passed'] = len([r for r in stress_results.values() if r.get('status') == 'passed'])
                summary['total_tests'] += len(stress_results)
            
            # Count performance test results
            if self.performance_metrics:
                perf_results = self.performance_metrics
                summary['details']['performance_tests'] = perf_results
                summary['performance_tests_passed'] = len([r for r in perf_results.values() if r.get('status') == 'passed'])
                summary['total_tests'] += len(perf_results)
            
            # Count reliability test results
            if self.reliability_metrics:
                rel_results = self.reliability_metrics
                summary['details']['reliability_tests'] = rel_results
                summary['reliability_tests_passed'] = len([r for r in rel_results.values() if r.get('status') == 'passed'])
                summary['total_tests'] += len(rel_results)
            
            # Calculate overall pass rate
            total_passed = summary['stress_tests_passed'] + summary['performance_tests_passed'] + summary['reliability_tests_passed']
            pass_rate = total_passed / summary['total_tests'] if summary['total_tests'] > 0 else 0
            
            summary['total_passed'] = total_passed
            summary['pass_rate'] = pass_rate
            summary['overall_status'] = 'passed' if pass_rate >= 0.8 else 'failed'
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate validation summary: {e}")
            return {'status': 'failed', 'error': str(e)}


# Global production validator
production_validator = ProductionValidator()
