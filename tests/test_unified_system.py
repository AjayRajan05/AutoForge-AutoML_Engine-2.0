"""
🧪 AutoForge Unified System Tests
Comprehensive testing of the unified AutoML system
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Import AutoForge components
from core.unified_automl import UnifiedAutoML
from core.estimator import AutoForgeRegressor, AutoForgeClassifier
from input_output.input_types import (
    AutoMLInput,
    get_test_input,
    get_small_test_input,
    get_text_test_input,
    get_time_series_test_input
)
from input_output.input_validator import InputValidator
from intelligence.intelligence_engine import IntelligenceEngine
from persistence.model_saver import ModelSaver


class TestUnifiedAutoML(unittest.TestCase):
    """Test the unified AutoML system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_saver = ModelSaver(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_automl_workflow(self):
        """Test basic AutoML workflow with synthetic data"""
        # Get test data
        test_input = get_test_input()
        
        # Create and train AutoML
        automl = UnifiedAutoML()
        automl.fit(test_input, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
        
        # Check that model is trained
        self.assertTrue(automl.is_fitted)
        self.assertIsNotNone(automl.best_model)
        self.assertIsNotNone(automl.best_score)
        
        # Test predictions
        X_test = test_input.get_features().head(10)
        predictions = automl.predict(X_test)
        
        self.assertEqual(len(predictions), 10)
        self.assertIsNotNone(automl.current_profile)
        
        # Test performance stats
        stats = automl.get_performance_stats()
        self.assertIn('best_score', stats)
        self.assertIn('training_time', stats)
        self.assertIn('model_type', stats)
        
        print(f"✅ Basic workflow test passed - Score: {stats['best_score']:.4f}")

    def test_task_type_propagation_regression(self):
        """Regression datasets should report task_type=regression in stats."""
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=200, n_features=6, noise=0.1, random_state=42)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df['target'] = y
        automl_input = AutoMLInput(data=df, target_column='target')

        automl = UnifiedAutoML()
        automl.fit(automl_input, enable_optimization=False, enable_tracking=False, enable_monitoring=False)

        stats = automl.get_performance_stats()
        self.assertEqual(stats['task_type'], 'regression')
        self.assertEqual(automl.task_type_, 'regression')
        comparison = automl.get_model_comparison()
        self.assertGreater(len(comparison), 0)

    def test_sklearn_estimator_api(self):
        """AutoForgeRegressor supports fit(X, y) and score."""
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        X, y = make_regression(n_samples=150, n_features=5, random_state=1)
        X = pd.DataFrame(X)
        y = pd.Series(y, name='target')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        model = AutoForgeRegressor(model_family='ml', max_trials=3, enable_optimization=False)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        self.assertIsNotNone(score)
        self.assertEqual(model.automl_.task_type_, 'regression')
    
    def test_small_dataset_workflow(self):
        """Test workflow with small dataset"""
        test_input = get_small_test_input()
        
        automl = UnifiedAutoML()
        automl.fit(test_input, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
        
        self.assertTrue(automl.is_fitted)
        stats = automl.get_performance_stats()
        
        # Should work with small datasets
        self.assertGreater(stats['best_score'], 0)
        
        print(f"✅ Small dataset test passed - Score: {stats['best_score']:.4f}")
    
    def test_text_data_workflow(self):
        """Test workflow with text data"""
        test_input = get_text_test_input()
        
        automl = UnifiedAutoML()
        automl.fit(test_input, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
        
        self.assertTrue(automl.is_fitted)
        stats = automl.get_performance_stats()
        
        # Should handle text data
        self.assertEqual(stats['data_type'], 'text')
        
        print(f"✅ Text data test passed - Score: {stats['best_score']:.4f}")
    
    def test_time_series_workflow(self):
        """Test workflow with time series data"""
        test_input = get_time_series_test_input()
        
        automl = UnifiedAutoML()
        automl.fit(test_input, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
        
        self.assertTrue(automl.is_fitted)
        stats = automl.get_performance_stats()
        
        # Should handle time series data
        self.assertEqual(stats['data_type'], 'time_series')
        
        print(f"✅ Time series test passed - Score: {stats['best_score']:.4f}")
    
    def test_model_persistence(self):
        """Test model saving and loading"""
        # Train model
        test_input = get_test_input()
        automl = UnifiedAutoML()
        automl.fit(test_input, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
        
        # Save model
        model_name = "test_model"
        model_path = automl.save_model(model_name)
        
        self.assertTrue(Path(model_path).exists())
        
        # Create new instance and load model
        automl2 = UnifiedAutoML()
        automl2.load_model(model_name)
        
        self.assertTrue(automl2.is_fitted)
        self.assertIsNotNone(automl2.best_model)
        
        # Test predictions with loaded model
        X_test = test_input.get_features().head(5)
        pred1 = automl.predict(X_test)
        pred2 = automl2.predict(X_test)
        
        np.testing.assert_array_equal(pred1, pred2)
        
        print(f"✅ Model persistence test passed")
    
    def test_input_validation(self):
        """Test input validation"""
        validator = InputValidator()
        
        # Test valid input
        test_input = get_test_input()
        result = validator.validate(test_input)
        self.assertTrue(result.is_valid)
        
        # Test invalid input (empty data cannot construct AutoMLInput)
        empty_data = pd.DataFrame({'target': []})
        with self.assertRaises(ValueError):
            AutoMLInput(data=empty_data, target_column='target')
        
        print(f"✅ Input validation test passed")
    
    def test_intelligence_engine(self):
        """Test intelligence engine analysis"""
        engine = IntelligenceEngine()
        
        # Test with synthetic data
        test_input = get_test_input()
        X = test_input.get_features()
        y = test_input.get_target()
        
        profile = engine.analyze(X, y)
        
        self.assertIsNotNone(profile)
        self.assertIsNotNone(profile.size_profile)
        self.assertIsNotNone(profile.quality_profile)
        self.assertIsNotNone(profile.data_type)
        self.assertGreater(profile.confidence, 0)
        
        # Test quick analysis
        quick_analysis = engine.get_quick_analysis(X, y)
        self.assertIn('shape', quick_analysis)
        self.assertIn('target_type', quick_analysis)
        
        # Test summary
        summary = engine.get_analysis_summary(profile)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        
        print(f"✅ Intelligence engine test passed - Data type: {profile.data_type}")
    
    def test_error_handling(self):
        """Test bulletproof error handling"""
        from core.bulletproof_error_handler import BulletproofErrorHandler
        
        handler = BulletproofErrorHandler()
        
        # Test error classification
        test_errors = [
            ValueError("Invalid parameter"),
            TypeError("Wrong data type"),
            MemoryError("Out of memory"),
            ImportError("Module not found")
        ]
        
        for error in test_errors:
            error_type = handler.classify_error(error)
            self.assertIsInstance(error_type, str)
            self.assertGreater(len(error_type), 0)
        
        print(f"✅ Error handling test passed")
    
    def test_config_options(self):
        """Test different configuration options"""
        test_input = get_test_input()
        
        # Test with custom config
        config = {
            'model_save_path': self.temp_dir,
            'auto_save_model': True,
            'random_state': 123
        }
        
        automl = UnifiedAutoML(config)
        automl.fit(test_input, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
        
        self.assertTrue(automl.is_fitted)
        self.assertEqual(automl.config['random_state'], 123)
        
        print(f"✅ Configuration options test passed")
    
    def test_explainability(self):
        """Test model explainability"""
        test_input = get_test_input()
        
        automl = UnifiedAutoML()
        automl.fit(test_input, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
        
        # Test explanations
        explanations = automl.explain()
        self.assertIsInstance(explanations, dict)
        self.assertIn('model_summary', explanations)
        
        # Test feature importance
        feature_importance = automl.get_feature_importance()
        if feature_importance:  # May not be available for all models
            self.assertIsInstance(feature_importance, dict)
            self.assertGreater(len(feature_importance), 0)
        
        print(f"✅ Explainability test passed")
    
    def test_different_preferences(self):
        """Test different user preferences"""
        test_input = get_test_input()
        
        preferences = ['auto', 'fast', 'accurate', 'robust']
        
        for preference in preferences:
            test_input_pref = AutoMLInput(
                data=test_input.data.copy(),
                target_column=test_input.target_column,
                user_preference=preference
            )
            
            automl = UnifiedAutoML()
            automl.fit(test_input_pref, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
            
            self.assertTrue(automl.is_fitted)
            stats = automl.get_performance_stats()
            
            print(f"  Preference '{preference}': Score {stats['best_score']:.4f}")
        
        print(f"✅ Different preferences test passed")


class TestInputTypes(unittest.TestCase):
    """Test input types and validation"""
    
    def test_automl_input_validation(self):
        """Test AutoMLInput validation"""
        # Valid input
        data = pd.DataFrame({
            'feature1': list(range(12)),
            'feature2': ['A', 'B', 'A', 'C', 'B', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
            'target': [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
        })
        
        input_data = AutoMLInput(
            data=data,
            target_column='target',
            task_type='classification'
        )
        
        self.assertEqual(input_data.get_shape(), (12, 3))
        self.assertEqual(len(input_data.get_features().columns), 2)
        self.assertEqual(len(input_data.get_target()), 12)
        
        # Test info
        info = input_data.get_info()
        self.assertIn('shape', info)
        self.assertIn('target_column', info)
        
        print(f"✅ AutoMLInput validation test passed")
    
    def test_test_inputs(self):
        """Test various test input generators"""
        # Small test input
        small_input = get_small_test_input()
        self.assertLess(small_input.get_shape()[0], 200)
        
        # Regular test input
        test_input = get_test_input()
        self.assertGreater(test_input.get_shape()[0], 500)
        
        # Text test input
        text_input = get_text_test_input()
        self.assertEqual(text_input.data_type, 'text')
        
        # Time series test input
        ts_input = get_time_series_test_input()
        self.assertEqual(ts_input.data_type, 'time_series')
        
        print(f"✅ Test input generators test passed")


def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🧪 Starting AutoForge Comprehensive Test Suite")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestUnifiedAutoML))
    suite.addTest(unittest.makeSuite(TestInputTypes))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"🧪 Test Results Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    if result.wasSuccessful():
        print(f"\n🎉 All tests passed! AutoForge is working correctly!")
    else:
        print(f"\n⚠️  Some tests failed. Please check the implementation.")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    run_comprehensive_test()
