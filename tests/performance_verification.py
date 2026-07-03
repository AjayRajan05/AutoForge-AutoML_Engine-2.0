"""
📊 Performance Verification - Compare AutoForge with existing systems
"""

import unittest
import pandas as pd
import numpy as np
import time
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PerformanceVerification:
    """
    Performance verification system for AutoForge
    """
    
    def __init__(self):
        """Initialize performance verification"""
        self.results = {}
        self.test_datasets = self._create_test_datasets()
        
    def _create_test_datasets(self) -> dict:
        """Create test datasets for performance comparison"""
        datasets = {}
        
        # Small classification dataset
        X_small, y_small = make_classification(n_samples=100, n_features=10, n_informative=5, 
                                             n_redundant=2, random_state=42)
        datasets['small_classification'] = {
            'X': pd.DataFrame(X_small, columns=[f'feature_{i}' for i in range(X_small.shape[1])]),
            'y': pd.Series(y_small, name='target'),
            'task_type': 'classification'
        }
        
        # Medium classification dataset
        X_medium, y_medium = make_classification(n_samples=500, n_features=20, n_informative=10, 
                                                n_redundant=5, random_state=42)
        datasets['medium_classification'] = {
            'X': pd.DataFrame(X_medium, columns=[f'feature_{i}' for i in range(X_medium.shape[1])]),
            'y': pd.Series(y_medium, name='target'),
            'task_type': 'classification'
        }
        
        # Small regression dataset
        X_reg_small, y_reg_small = make_regression(n_samples=100, n_features=10, n_informative=5, 
                                                  noise=0.1, random_state=42)
        datasets['small_regression'] = {
            'X': pd.DataFrame(X_reg_small, columns=[f'feature_{i}' for i in range(X_reg_small.shape[1])]),
            'y': pd.Series(y_reg_small, name='target'),
            'task_type': 'regression'
        }
        
        # Medium regression dataset
        X_reg_medium, y_reg_medium = make_regression(n_samples=500, n_features=20, n_informative=10, 
                                                   noise=0.1, random_state=42)
        datasets['medium_regression'] = {
            'X': pd.DataFrame(X_reg_medium, columns=[f'feature_{i}' for i in range(X_reg_medium.shape[1])]),
            'y': pd.Series(y_reg_medium, name='target'),
            'task_type': 'regression'
        }
        
        return datasets
    
    def test_autoforge_performance(self, dataset_name: str) -> dict:
        """Test AutoForge performance on a dataset"""
        print(f"\n🚀 Testing AutoForge on {dataset_name}...")
        
        dataset = self.test_datasets[dataset_name]
        X, y = dataset['X'], dataset['y']
        task_type = dataset['task_type']
        
        try:
            # Import AutoForge
            from autoforge.core.unified_automl import UnifiedAutoML
            from autoforge.input_output.input_types import AutoMLInput
            
            # Create AutoML input
            automl_input = AutoMLInput(
                data=pd.concat([X, y], axis=1),
                target_column='target',
                max_trials=10,  # Moderate for testing
                max_time=60      # 1 minute for testing
            )
            
            # Initialize and fit AutoForge
            autoforge = UnifiedAutoML()
            
            start_time = time.time()
            autoforge.fit(automl_input, enable_tracking=False, enable_monitoring=False, enable_optimization=False)
            training_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            predictions = autoforge.predict(X)
            prediction_time = time.time() - start_time
            
            # Calculate performance metrics
            if task_type == 'classification':
                accuracy = accuracy_score(y, predictions)
                f1 = f1_score(y, predictions, average='weighted')
                metrics = {'accuracy': accuracy, 'f1_score': f1}
                primary_score = accuracy
            else:
                r2 = r2_score(y, predictions)
                mse = mean_squared_error(y, predictions)
                metrics = {'r2_score': r2, 'mse': mse}
                primary_score = r2
            
            result = {
                'system': 'AutoForge',
                'dataset': dataset_name,
                'task_type': task_type,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'total_time': training_time + prediction_time,
                'primary_score': primary_score,
                'metrics': metrics,
                'model_type': type(autoforge.best_model).__name__,
                'features_used': len(autoforge.training_metadata.get('selected_features', [])),
                'status': 'success'
            }
            
            print(f"✅ AutoForge: {primary_score:.4f} score in {training_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"❌ AutoForge failed: {e}")
            return {
                'system': 'AutoForge',
                'dataset': dataset_name,
                'task_type': task_type,
                'status': 'failed',
                'error': str(e)
            }
    
    def test_baseline_performance(self, dataset_name: str) -> dict:
        """Test baseline model performance"""
        print(f"📊 Testing baseline models on {dataset_name}...")
        
        dataset = self.test_datasets[dataset_name]
        X, y = dataset['X'], dataset['y']
        task_type = dataset['task_type']
        
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.model_selection import cross_val_score
            
            # Choose baseline model
            if task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                scoring = 'accuracy'
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                scoring = 'r2'
            
            # Time the training
            start_time = time.time()
            model.fit(X, y)
            training_time = time.time() - start_time
            
            # Time the prediction
            start_time = time.time()
            predictions = model.predict(X)
            prediction_time = time.time() - start_time
            
            # Calculate performance metrics
            if task_type == 'classification':
                accuracy = accuracy_score(y, predictions)
                f1 = f1_score(y, predictions, average='weighted')
                metrics = {'accuracy': accuracy, 'f1_score': f1}
                primary_score = accuracy
            else:
                r2 = r2_score(y, predictions)
                mse = mean_squared_error(y, predictions)
                metrics = {'r2_score': r2, 'mse': mse}
                primary_score = r2
            
            result = {
                'system': 'Baseline_RandomForest',
                'dataset': dataset_name,
                'task_type': task_type,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'total_time': training_time + prediction_time,
                'primary_score': primary_score,
                'metrics': metrics,
                'model_type': 'RandomForest',
                'status': 'success'
            }
            
            print(f"✅ Baseline: {primary_score:.4f} score in {training_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"❌ Baseline failed: {e}")
            return {
                'system': 'Baseline_RandomForest',
                'dataset': dataset_name,
                'task_type': task_type,
                'status': 'failed',
                'error': str(e)
            }
    
    def test_existing_systems(self, dataset_name: str) -> list:
        """Test existing AutoML systems if available"""
        print(f"🔗 Testing existing systems on {dataset_name}...")
        
        dataset = self.test_datasets[dataset_name]
        X, y = dataset['X'], dataset['y']
        task_type = dataset['task_type']
        
        results = []
        
        # Test existing API systems
        try:
            from autoforge.api.api_integration import api_integrator
            
            if api_integrator.available_systems.get('automl', False):
                try:
                    from ..api.automl import AutoML as ExistingAutoML
                    
                    model = ExistingAutoML(max_trials=10, random_state=42)
                    
                    start_time = time.time()
                    model.fit(X, y)
                    training_time = time.time() - start_time
                    
                    start_time = time.time()
                    predictions = model.predict(X)
                    prediction_time = time.time() - start_time
                    
                    if task_type == 'classification':
                        accuracy = accuracy_score(y, predictions)
                        primary_score = accuracy
                    else:
                        r2 = r2_score(y, predictions)
                        primary_score = r2
                    
                    result = {
                        'system': 'Existing_AutoML',
                        'dataset': dataset_name,
                        'task_type': task_type,
                        'training_time': training_time,
                        'prediction_time': prediction_time,
                        'total_time': training_time + prediction_time,
                        'primary_score': primary_score,
                        'status': 'success'
                    }
                    
                    results.append(result)
                    print(f"✅ Existing AutoML: {primary_score:.4f} score in {training_time:.2f}s")
                    
                except Exception as e:
                    print(f"⚠️ Existing AutoML failed: {e}")
        except ImportError:
            print("⚠️ API integration not available")
        
        return results
    
    def run_performance_comparison(self) -> dict:
        """Run comprehensive performance comparison"""
        print("📊 Starting Performance Verification...")
        print("=" * 60)
        
        all_results = []
        
        for dataset_name in self.test_datasets.keys():
            print(f"\n🎯 Testing dataset: {dataset_name}")
            print("-" * 40)
            
            # Test AutoForge
            autoforge_result = self.test_autoforge_performance(dataset_name)
            all_results.append(autoforge_result)
            
            # Test baseline
            baseline_result = self.test_baseline_performance(dataset_name)
            all_results.append(baseline_result)
            
            # Test existing systems
            existing_results = self.test_existing_systems(dataset_name)
            all_results.extend(existing_results)
        
        # Analyze results
        analysis = self._analyze_results(all_results)
        
        return {
            'results': all_results,
            'analysis': analysis,
            'summary': self._create_summary(all_results, analysis)
        }
    
    def _analyze_results(self, results: list) -> dict:
        """Analyze performance results"""
        analysis = {
            'datasets': {},
            'overall_performance': {},
            'improvements': {}
        }
        
        # Group results by dataset
        for result in results:
            if result['status'] == 'success':
                dataset = result['dataset']
                if dataset not in analysis['datasets']:
                    analysis['datasets'][dataset] = []
                analysis['datasets'][dataset].append(result)
        
        # Analyze each dataset
        for dataset, dataset_results in analysis['datasets'].items():
            # Find AutoForge result
            autoforge_result = next((r for r in dataset_results if r['system'] == 'AutoForge'), None)
            baseline_result = next((r for r in dataset_results if r['system'] == 'Baseline_RandomForest'), None)
            
            if autoforge_result and baseline_result:
                # Calculate improvement
                score_improvement = ((autoforge_result['primary_score'] - baseline_result['primary_score']) / 
                                   abs(baseline_result['primary_score'])) * 100
                
                time_improvement = ((baseline_result['total_time'] - autoforge_result['total_time']) / 
                                  baseline_result['total_time']) * 100
                
                analysis['improvements'][dataset] = {
                    'score_improvement_percent': score_improvement,
                    'time_improvement_percent': time_improvement,
                    'autoforge_better': score_improvement > 0
                }
        
        return analysis
    
    def _create_summary(self, results: list, analysis: dict) -> dict:
        """Create performance summary"""
        successful_results = [r for r in results if r['status'] == 'success']
        autoforge_results = [r for r in successful_results if r['system'] == 'AutoForge']
        baseline_results = [r for r in successful_results if r['system'] == 'Baseline_RandomForest']
        
        summary = {
            'total_tests': len(results),
            'successful_tests': len(successful_results),
            'autoforge_tests': len(autoforge_results),
            'baseline_tests': len(baseline_results),
            'average_autoforge_score': np.mean([r['primary_score'] for r in autoforge_results]) if autoforge_results else 0,
            'average_baseline_score': np.mean([r['primary_score'] for r in baseline_results]) if baseline_results else 0,
            'average_autoforge_time': np.mean([r['total_time'] for r in autoforge_results]) if autoforge_results else 0,
            'average_baseline_time': np.mean([r['total_time'] for r in baseline_results]) if baseline_results else 0,
            'overall_improvement': 0
        }
        
        if summary['average_autoforge_score'] > 0 and summary['average_baseline_score'] > 0:
            summary['overall_improvement'] = ((summary['average_autoforge_score'] - summary['average_baseline_score']) / 
                                             summary['average_baseline_score']) * 100
        
        return summary
    
    def generate_report(self, results: dict) -> str:
        """Generate performance verification report"""
        lines = []
        
        # Header
        lines.append("📊 AutoForge Performance Verification Report")
        lines.append("=" * 60)
        
        # Summary
        summary = results['summary']
        lines.append("\n📈 Summary:")
        lines.append(f"  Total Tests: {summary['total_tests']}")
        lines.append(f"  Successful Tests: {summary['successful_tests']}")
        lines.append(f"  AutoForge Avg Score: {summary['average_autoforge_score']:.4f}")
        lines.append(f"  Baseline Avg Score: {summary['average_baseline_score']:.4f}")
        lines.append(f"  Overall Improvement: {summary['overall_improvement']:.2f}%")
        
        # Dataset improvements
        lines.append("\n🎯 Dataset-by-Dataset Results:")
        for dataset, improvement in results['analysis']['improvements'].items():
            lines.append(f"  {dataset}:")
            lines.append(f"    Score Improvement: {improvement['score_improvement_percent']:.2f}%")
            lines.append(f"    Time Improvement: {improvement['time_improvement_percent']:.2f}%")
            lines.append(f"    AutoForge Better: {'✅ Yes' if improvement['autoforge_better'] else '❌ No'}")
        
        # Conclusion
        lines.append("\n🎉 Conclusion:")
        if summary['overall_improvement'] > 0:
            lines.append(f"  AutoForge outperforms baseline by {summary['overall_improvement']:.2f}%")
            lines.append("  ✅ Performance verification PASSED")
        else:
            lines.append(f"  AutoForge underperforms baseline by {abs(summary['overall_improvement']):.2f}%")
            lines.append("  ⚠️ Performance verification FAILED")
        
        return "\n".join(lines)


def run_performance_verification():
    """Run performance verification"""
    verifier = PerformanceVerification()
    results = verifier.run_performance_comparison()
    
    # Generate report
    report = verifier.generate_report(results)
    print(report)
    
    return results


if __name__ == '__main__':
    run_performance_verification()
