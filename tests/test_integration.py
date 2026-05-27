"""
🧪 Integration Tests - Test all AutoForge integrations
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_automl import UnifiedAutoML
from input_output.input_types import AutoMLInput


class TestIntegration(unittest.TestCase):
    """Comprehensive integration tests for AutoForge"""
    
    def setUp(self):
        """Set up test data"""
        # Create small classification dataset
        X_cls, y_cls = make_classification(n_samples=100, n_features=10, n_informative=5, 
                                         n_redundant=2, random_state=42)
        self.X_cls = pd.DataFrame(X_cls, columns=[f'feature_{i}' for i in range(X_cls.shape[1])])
        self.y_cls = pd.Series(y_cls, name='target')
        
        # Create small regression dataset
        X_reg, y_reg = make_regression(n_samples=100, n_features=10, n_informative=5, 
                                      noise=0.1, random_state=42)
        self.X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(X_reg.shape[1])])
        self.y_reg = pd.Series(y_reg, name='target')
        
        # Initialize AutoForge (fast search for integration tests)
        self.autoforge = UnifiedAutoML({'search_depth': 'fast'})
    
    def test_basic_classification_integration(self):
        """Test basic classification integration"""
        print("\n🧪 Testing basic classification integration...")
        
        # Create AutoML input
        automl_input = AutoMLInput(
            data=pd.concat([self.X_cls, self.y_cls], axis=1),
            target_column='target',
            search_depth='fast',
            max_trials=5,
            max_time=30
        )
        
        # Fit AutoForge
        self.autoforge.fit(automl_input, enable_tracking=False, enable_monitoring=False, enable_optimization=False)
        
        # Check that model is fitted
        self.assertTrue(self.autoforge.is_fitted)
        self.assertIsNotNone(self.autoforge.best_model)
        self.assertIsNotNone(self.autoforge.best_score)
        
        # Test prediction
        predictions = self.autoforge.predict(self.X_cls)
        self.assertEqual(len(predictions), len(self.X_cls))
        
        # Test explainability
        explanations = self.autoforge.explain()
        self.assertIsInstance(explanations, dict)
        
        print("✅ Basic classification integration test passed")
    
    def test_basic_regression_integration(self):
        """Test basic regression integration"""
        print("\n🧪 Testing basic regression integration...")
        
        # Create AutoML input
        automl_input = AutoMLInput(
            data=pd.concat([self.X_reg, self.y_reg], axis=1),
            target_column='target',
            search_depth='fast',
            max_trials=5,
            max_time=30
        )
        
        # Fit AutoForge
        self.autoforge.fit(automl_input, enable_tracking=False, enable_monitoring=False, enable_optimization=False)
        
        # Check that model is fitted
        self.assertTrue(self.autoforge.is_fitted)
        self.assertIsNotNone(self.autoforge.best_model)
        self.assertIsNotNone(self.autoforge.best_score)
        
        # Test prediction
        predictions = self.autoforge.predict(self.X_reg)
        self.assertEqual(len(predictions), len(self.X_reg))
        
        print("✅ Basic regression integration test passed")
    
    def test_feature_registry_integration(self):
        """Test feature registry integration"""
        print("\n🧪 Testing feature registry integration...")
        
        # Test feature registry
        from registry.feature_registry import feature_registry
        
        # List available features
        available_features = feature_registry.list_features()
        self.assertIsInstance(available_features, list)
        
        # Get registry summary
        summary = feature_registry.get_registry_summary()
        self.assertIn('total_features', summary)
        
        print("✅ Feature registry integration test passed")
    
    def test_model_registry_integration(self):
        """Test model registry integration"""
        print("\n🧪 Testing model registry integration...")
        
        # Test model registry
        from registry.model_registry import model_registry
        
        # List available models
        available_models = model_registry.get_available_models()
        self.assertIsInstance(available_models, dict)
        self.assertTrue(len(available_models) > 0)

        recommendations = model_registry.recommend_models(100, 10, 'classification')
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

        model = model_registry.get_model('random_forest', 'classification')
        self.assertIsNotNone(model)
        
        print("✅ Model registry integration test passed")
    
    def test_processor_integration(self):
        """Test processor integration"""
        print("\n🧪 Testing processor integration...")
        
        # Test processor integrator
        from processors.processor_integration import processor_integrator
        
        # Test data type detection
        data_type = processor_integrator.detect_data_type(self.X_cls, self.y_cls)
        self.assertIn(data_type, ['tabular', 'text', 'image', 'time_series'])
        
        # Test data processing
        X_processed, y_processed = processor_integrator.process_data(self.X_cls, self.y_cls)
        self.assertIsInstance(X_processed, pd.DataFrame)
        self.assertEqual(len(X_processed), len(self.X_cls))
        
        print("✅ Processor integration test passed")
    
    def test_ensemble_integration(self):
        """Test ensemble integration"""
        print("\n🧪 Testing ensemble integration...")
        
        # Test ensemble integrator
        from ensemble.ensemble_integration import ensemble_integrator
        
        # Test voting ensemble
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        models = [RandomForestClassifier(n_estimators=5, random_state=42),
                 LogisticRegression(random_state=42, max_iter=100)]
        for model in models:
            model.fit(self.X_cls, self.y_cls)

        ensemble = ensemble_integrator.create_voting_ensemble(models, 'classification')
        self.assertIsNotNone(ensemble)
        ensemble.fit(self.X_cls, self.y_cls)

        results = ensemble_integrator.evaluate_ensemble(ensemble, self.X_cls, self.y_cls, models)
        self.assertIn('ensemble_score', results)
        
        print("✅ Ensemble integration test passed")
    
    def test_api_integration(self):
        """Test API integration"""
        print("\n🧪 Testing API integration...")

        from api.api_integration import api_integrator

        status = api_integrator.get_status()
        self.assertIsInstance(status, dict)
        self.assertIn('is_fitted', status)

        print("✅ API integration test passed")
    
    def test_benchmarking_integration(self):
        """Test benchmarking integration"""
        print("\n🧪 Testing benchmarking integration...")
        
        # Test benchmarking integrator
        from benchmarking.benchmark_integration import benchmarking_integrator
        
        # Check available benchmarkers
        available_benchmarkers = benchmarking_integrator.available_benchmarkers
        self.assertIsInstance(available_benchmarkers, dict)
        
        print("✅ Benchmarking integration test passed")
    
    def test_systemization_integration(self):
        """Test systemization integration"""
        print("\n🧪 Testing systemization integration...")
        
        # Test systemization integrator
        from systemization.systemization_integration import systemization_integrator
        
        # Check available components
        available_components = systemization_integrator.available_components
        self.assertIsInstance(available_components, dict)
        
        print("✅ Systemization integration test passed")
    
    def test_tracking_integration(self):
        """Test tracking integration"""
        print("\n🧪 Testing tracking integration...")
        
        # Test tracking integrator
        from tracking.tracking_integration import tracking_integrator
        
        # Check available components
        available_components = tracking_integrator.available_components
        self.assertIsInstance(available_components, dict)
        
        print("✅ Tracking integration test passed")
    
    def test_optimizer_integration(self):
        """Test optimizer integration"""
        print("\n🧪 Testing optimizer integration...")
        
        # Test optimizer integrator
        from optimizer.optimizer_integration import optimizer_integrator
        
        # Check available optimizers
        available_optimizers = optimizer_integrator.available_optimizers
        self.assertIsInstance(available_optimizers, dict)
        
        print("✅ Optimizer integration test passed")
    
    def test_full_integration_summary(self):
        """Test full integration summary"""
        print("\n🧪 Testing full integration summary...")
        
        # Get integration summary
        summary = self.autoforge.get_integration_summary()
        
        # Check that all integrations are present
        expected_keys = [
            'model_registry', 'feature_registry', 'processors',
            'ensemble', 'optimizer', 'api',
        ]
        
        for key in expected_keys:
            self.assertIn(key, summary)
        
        print("✅ Full integration summary test passed")
    
    def test_model_persistence_integration(self):
        """Test model persistence integration"""
        print("\n🧪 Testing model persistence integration...")
        
        # Create and train a model
        automl_input = AutoMLInput(
            data=pd.concat([self.X_cls, self.y_cls], axis=1),
            target_column='target',
            max_trials=3,
            max_time=20
        )
        
        self.autoforge.fit(automl_input, enable_tracking=False, enable_monitoring=False, enable_optimization=False)
        
        # Save model
        model_path = self.autoforge.save_model('test_integration_model')
        self.assertIsNotNone(model_path)
        
        # Create new instance and load model
        new_autoforge = UnifiedAutoML()
        new_autoforge.load_model('test_integration_model')
        
        # Test that loaded model works
        self.assertTrue(new_autoforge.is_fitted)
        predictions = new_autoforge.predict(self.X_cls)
        self.assertEqual(len(predictions), len(self.X_cls))
        
        print("✅ Model persistence integration test passed")
    
    def test_cli_integration(self):
        """Test CLI integration"""
        print("\n🧪 Testing CLI integration...")

        from cli.cli_integration import CLIIntegrator

        integrator = CLIIntegrator()
        self.assertIsNotNone(integrator)

        print("✅ CLI integration test passed")


def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Starting AutoForge Integration Tests...")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Integration Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 All integration tests passed!")
        return True
    else:
        print("\n⚠️  Some integration tests failed!")
        return False


if __name__ == '__main__':
    run_integration_tests()
