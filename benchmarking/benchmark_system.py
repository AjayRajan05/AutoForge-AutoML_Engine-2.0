"""
Benchmarking System
Compare AutoForge with Auto-sklearn and H2O AutoML
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

# Import our AutoML system
from .automl import AutoML

# Try to import competing AutoML systems
try:
    import autosklearn
    AUTOSKLEARN_AVAILABLE = True
except ImportError:
    AUTOSKLEARN_AVAILABLE = False
    autosklearn = None

try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False
    h2o = None
    H2OAutoML = None

logger = logging.getLogger(__name__)


class BenchmarkSystem:
    """
    Comprehensive benchmarking system for AutoML comparison
    """
    
    def __init__(self, 
                 max_runtime_per_model: int = 300,  # 5 minutes per model
                 n_trials_autoforge: int = 50,
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize benchmark system
        
        Args:
            max_runtime_per_model: Maximum runtime in seconds per model
            n_trials_autoforge: Number of trials for AutoForge
            test_size: Test set size ratio
            random_state: Random seed
        """
        self.max_runtime_per_model = max_runtime_per_model
        self.n_trials_autoforge = n_trials_autoforge
        self.test_size = test_size
        self.random_state = random_state
        
        # Results storage
        self.results = {}
        self.dataset_info = {}
        
        # Initialize H2O if available
        if H2O_AVAILABLE:
            try:
                h2o.init(max_mem_size="2G")
                logger.info("H2O initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize H2O: {e}")
                H2O_AVAILABLE = False
    
    def create_classification_datasets(self) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create classification benchmark datasets
        
        Returns:
            List of (dataset_name, X_train, X_test, y_train, y_test)
        """
        datasets = []
        
        # Dataset 1: Simple binary classification
        X1, y1 = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=2,
            random_state=self.random_state
        )
        X1_train, X1_test, y1_train, y1_test = train_test_split(
            X1, y1, test_size=self.test_size, random_state=self.random_state
        )
        datasets.append(("simple_binary", X1_train, X1_test, y1_train, y1_test))
        
        # Dataset 2: Imbalanced binary classification
        X2, y2 = make_classification(
            n_samples=2000,
            n_features=15,
            n_informative=8,
            n_redundant=3,
            n_clusters_per_class=2,
            weights=[0.9, 0.1],  # 90:10 imbalance
            random_state=self.random_state
        )
        X2_train, X2_test, y2_train, y2_test = train_test_split(
            X2, y2, test_size=self.test_size, random_state=self.random_state, stratify=y2
        )
        datasets.append(("imbalanced_binary", X2_train, X2_test, y2_train, y2_test))
        
        # Dataset 3: Multi-class classification
        X3, y3 = make_classification(
            n_samples=1500,
            n_features=25,
            n_informative=12,
            n_redundant=6,
            n_classes=5,
            n_clusters_per_class=1,
            random_state=self.random_state
        )
        X3_train, X3_test, y3_train, y3_test = train_test_split(
            X3, y3, test_size=self.test_size, random_state=self.random_state, stratify=y3
        )
        datasets.append(("multiclass", X3_train, X3_test, y3_train, y3_test))
        
        # Dataset 4: High-dimensional classification
        X4, y4 = make_classification(
            n_samples=800,
            n_features=100,
            n_informative=30,
            n_redundant=20,
            n_clusters_per_class=2,
            random_state=self.random_state
        )
        X4_train, X4_test, y4_train, y4_test = train_test_split(
            X4, y4, test_size=self.test_size, random_state=self.random_state
        )
        datasets.append(("high_dimensional", X4_train, X4_test, y4_train, y4_test))
        
        return datasets
    
    def create_regression_datasets(self) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create regression benchmark datasets
        
        Returns:
            List of (dataset_name, X_train, X_test, y_train, y_test)
        """
        datasets = []
        
        # Dataset 1: Simple regression
        X1, y1 = make_regression(
            n_samples=1000,
            n_features=15,
            n_informative=10,
            noise=0.1,
            random_state=self.random_state
        )
        X1_train, X1_test, y1_train, y1_test = train_test_split(
            X1, y1, test_size=self.test_size, random_state=self.random_state
        )
        datasets.append(("simple_regression", X1_train, X1_test, y1_train, y1_test))
        
        # Dataset 2: Complex regression with interactions
        X2, y2 = make_regression(
            n_samples=1500,
            n_features=20,
            n_informative=15,
            noise=0.2,
            random_state=self.random_state
        )
        X2_train, X2_test, y2_train, y2_test = train_test_split(
            X2, y2, test_size=self.test_size, random_state=self.random_state
        )
        datasets.append(("complex_regression", X2_train, X2_test, y2_train, y2_test))
        
        # Dataset 3: High-dimensional regression
        X3, y3 = make_regression(
            n_samples=800,
            n_features=80,
            n_informative=25,
            noise=0.15,
            random_state=self.random_state
        )
        X3_train, X3_test, y3_train, y3_test = train_test_split(
            X3, y3, test_size=self.test_size, random_state=self.random_state
        )
        datasets.append(("high_dim_regression", X3_train, X3_test, y3_train, y3_test))
        
        return datasets
    
    def benchmark_autoforge(self, 
                           X_train: np.ndarray, 
                           X_test: np.ndarray, 
                           y_train: np.ndarray, 
                           y_test: np.ndarray,
                           task_type: str) -> Dict[str, Any]:
        """
        Benchmark AutoForge AutoML
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            task_type: "classification" or "regression"
            
        Returns:
            Benchmark results
        """
        logger.info("Benchmarking AutoForge...")
        
        start_time = time.time()
        
        try:
            # Initialize AutoForge
            automl = AutoML(
                n_trials=self.n_trials_autoforge,
                timeout=self.max_runtime_per_model,
                use_adaptive_optimization=True,
                use_dataset_optimization=True,
                use_caching=True,
                show_progress=False
            )
            
            # Fit model
            automl.fit(X_train, y_train)
            
            # Make predictions
            y_pred = automl.predict(X_test)
            
            # Calculate metrics
            if task_type == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                metrics = {"accuracy": accuracy, "f1_score": f1}
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                metrics = {"mse": mse, "r2_score": r2}
            
            training_time = time.time() - start_time
            prediction_time = 0.0  # Not measured separately
            
            results = {
                "model_name": "AutoForge",
                "training_time": training_time,
                "prediction_time": prediction_time,
                "metrics": metrics,
                "status": "success",
                "error": None,
                "n_trials": automl.optimization_metadata.get("total_trials", self.n_trials_autoforge),
                "best_model": getattr(automl, 'best_model_name', 'unknown'),
                "dataset_optimization": automl.dataset_metadata.get("strategy", "none")
            }
            
            logger.info(f"AutoForge completed in {training_time:.2f}s")
            
        except Exception as e:
            logger.error(f"AutoForge failed: {e}")
            results = {
                "model_name": "AutoForge",
                "training_time": time.time() - start_time,
                "prediction_time": 0.0,
                "metrics": {},
                "status": "failed",
                "error": str(e),
                "n_trials": 0,
                "best_model": None,
                "dataset_optimization": None
            }
        
        return results
    
    def benchmark_autosklearn(self, 
                             X_train: np.ndarray, 
                             X_test: np.ndarray, 
                             y_train: np.ndarray, 
                             y_test: np.ndarray,
                             task_type: str) -> Dict[str, Any]:
        """
        Benchmark Auto-sklearn
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            task_type: "classification" or "regression"
            
        Returns:
            Benchmark results
        """
        if not AUTOSKLEARN_AVAILABLE:
            return {
                "model_name": "Auto-sklearn",
                "training_time": 0.0,
                "prediction_time": 0.0,
                "metrics": {},
                "status": "not_available",
                "error": "Auto-sklearn not installed",
                "n_trials": 0,
                "best_model": None,
                "dataset_optimization": None
            }
        
        logger.info("Benchmarking Auto-sklearn...")
        
        start_time = time.time()
        
        try:
            # Initialize Auto-sklearn
            if task_type == "classification":
                automl = autosklearn.classification.AutoSklearnClassifier(
                    time_left_for_this_task=self.max_runtime_per_model,
                    per_run_time_limit=60,
                    ensemble_size=1,  # Disable ensemble for fair comparison
                    seed=self.random_state
                )
            else:
                automl = autosklearn.regression.AutoSklearnRegressor(
                    time_left_for_this_task=self.max_runtime_per_model,
                    per_run_time_limit=60,
                    ensemble_size=1,
                    seed=self.random_state
                )
            
            # Fit model
            automl.fit(X_train, y_train)
            
            # Make predictions
            y_pred = automl.predict(X_test)
            
            # Calculate metrics
            if task_type == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                metrics = {"accuracy": accuracy, "f1_score": f1}
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                metrics = {"mse": mse, "r2_score": r2}
            
            training_time = time.time() - start_time
            
            results = {
                "model_name": "Auto-sklearn",
                "training_time": training_time,
                "prediction_time": 0.0,
                "metrics": metrics,
                "status": "success",
                "error": None,
                "n_trials": len(automl.automl_.models_) if hasattr(automl.automl_, 'models_') else 0,
                "best_model": str(automl.show_models().index[0]) if hasattr(automl, 'show_models') and len(automl.show_models()) > 0 else 'unknown',
                "dataset_optimization": "none"
            }
            
            logger.info(f"Auto-sklearn completed in {training_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Auto-sklearn failed: {e}")
            results = {
                "model_name": "Auto-sklearn",
                "training_time": time.time() - start_time,
                "prediction_time": 0.0,
                "metrics": {},
                "status": "failed",
                "error": str(e),
                "n_trials": 0,
                "best_model": None,
                "dataset_optimization": None
            }
        
        return results
    
    def benchmark_h2o_automl(self, 
                           X_train: np.ndarray, 
                           X_test: np.ndarray, 
                           y_train: np.ndarray, 
                           y_test: np.ndarray,
                           task_type: str) -> Dict[str, Any]:
        """
        Benchmark H2O AutoML
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            task_type: "classification" or "regression"
            
        Returns:
            Benchmark results
        """
        if not H2O_AVAILABLE:
            return {
                "model_name": "H2O AutoML",
                "training_time": 0.0,
                "prediction_time": 0.0,
                "metrics": {},
                "status": "not_available",
                "error": "H2O not installed",
                "n_trials": 0,
                "best_model": None,
                "dataset_optimization": None
            }
        
        logger.info("Benchmarking H2O AutoML...")
        
        start_time = time.time()
        
        try:
            # Convert to H2O frames
            train_df = pd.DataFrame(X_train)
            train_df['target'] = y_train
            train_h2o = h2o.H2OFrame(train_df)
            
            test_df = pd.DataFrame(X_test)
            test_df['target'] = y_test
            test_h2o = h2o.H2OFrame(test_df)
            
            # Define features and target
            feature_cols = [f"col_{i}" for i in range(X_train.shape[1])]
            target_col = "target"
            
            # Update column names
            for i, col in enumerate(feature_cols):
                train_h2o[col] = train_h2o[f"C{i}"]
                test_h2o[col] = test_h2o[f"C{i}"]
            
            # Initialize H2O AutoML
            aml = H2OAutoML(
                max_runtime_secs=self.max_runtime_per_model,
                seed=self.random_state,
                nfolds=3,  # 3-fold cross-validation
                exclude_algos=["DeepLearning"],  # Exclude deep learning for fair comparison
                verbosity='info'
            )
            
            # Train model
            aml.train(x=feature_cols, y=target_col, training_frame=train_h2o)
            
            # Make predictions
            predictions = aml.predict(test_h2o)
            y_pred = predictions['predict'].as_data_frame().values.flatten()
            
            # Convert back to numpy for metric calculation
            y_test_np = y_test
            
            # Calculate metrics
            if task_type == "classification":
                accuracy = accuracy_score(y_test_np, y_pred)
                f1 = f1_score(y_test_np, y_pred, average='weighted')
                metrics = {"accuracy": accuracy, "f1_score": f1}
            else:
                mse = mean_squared_error(y_test_np, y_pred)
                r2 = r2_score(y_test_np, y_pred)
                metrics = {"mse": mse, "r2_score": r2}
            
            training_time = time.time() - start_time
            
            results = {
                "model_name": "H2O AutoML",
                "training_time": training_time,
                "prediction_time": 0.0,
                "metrics": metrics,
                "status": "success",
                "error": None,
                "n_trials": len(aml.leaderboard.as_data_frame()),
                "best_model": aml.leader.model_id,
                "dataset_optimization": "none"
            }
            
            logger.info(f"H2O AutoML completed in {training_time:.2f}s")
            
        except Exception as e:
            logger.error(f"H2O AutoML failed: {e}")
            results = {
                "model_name": "H2O AutoML",
                "training_time": time.time() - start_time,
                "prediction_time": 0.0,
                "metrics": {},
                "status": "failed",
                "error": str(e),
                "n_trials": 0,
                "best_model": None,
                "dataset_optimization": None
            }
        
        return results
    
    def run_classification_benchmark(self) -> Dict[str, Any]:
        """
        Run complete classification benchmark
        
        Returns:
            Benchmark results
        """
        logger.info("Starting classification benchmark...")
        
        datasets = self.create_classification_datasets()
        results = {}
        
        for dataset_name, X_train, X_test, y_train, y_test in datasets:
            logger.info(f"Benchmarking dataset: {dataset_name}")
            
            # Store dataset info
            self.dataset_info[dataset_name] = {
                "n_samples_train": len(X_train),
                "n_samples_test": len(X_test),
                "n_features": X_train.shape[1],
                "task_type": "classification"
            }
            
            dataset_results = {}
            
            # Benchmark AutoForge
            dataset_results["autoforge"] = self.benchmark_autoforge(
                X_train, X_test, y_train, y_test, "classification"
            )
            
            # Benchmark Auto-sklearn
            dataset_results["autosklearn"] = self.benchmark_autosklearn(
                X_train, X_test, y_train, y_test, "classification"
            )
            
            # Benchmark H2O AutoML
            dataset_results["h2o"] = self.benchmark_h2o_automl(
                X_train, X_test, y_train, y_test, "classification"
            )
            
            results[dataset_name] = dataset_results
        
        return results
    
    def run_regression_benchmark(self) -> Dict[str, Any]:
        """
        Run complete regression benchmark
        
        Returns:
            Benchmark results
        """
        logger.info("Starting regression benchmark...")
        
        datasets = self.create_regression_datasets()
        results = {}
        
        for dataset_name, X_train, X_test, y_train, y_test in datasets:
            logger.info(f"Benchmarking dataset: {dataset_name}")
            
            # Store dataset info
            self.dataset_info[dataset_name] = {
                "n_samples_train": len(X_train),
                "n_samples_test": len(X_test),
                "n_features": X_train.shape[1],
                "task_type": "regression"
            }
            
            dataset_results = {}
            
            # Benchmark AutoForge
            dataset_results["autoforge"] = self.benchmark_autoforge(
                X_train, X_test, y_train, y_test, "regression"
            )
            
            # Benchmark Auto-sklearn
            dataset_results["autosklearn"] = self.benchmark_autosklearn(
                X_train, X_test, y_train, y_test, "regression"
            )
            
            # Benchmark H2O AutoML
            dataset_results["h2o"] = self.benchmark_h2o_automl(
                X_train, X_test, y_train, y_test, "regression"
            )
            
            results[dataset_name] = dataset_results
        
        return results
    
    def generate_benchmark_report(self, 
                                 classification_results: Dict[str, Any],
                                 regression_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive benchmark report
        
        Args:
            classification_results: Classification benchmark results
            regression_results: Regression benchmark results
            
        Returns:
            Benchmark report
        """
        report = {
            "summary": {},
            "classification": classification_results,
            "regression": regression_results,
            "dataset_info": self.dataset_info,
            "configuration": {
                "max_runtime_per_model": self.max_runtime_per_model,
                "n_trials_autoforge": self.n_trials_autoforge,
                "test_size": self.test_size,
                "random_state": self.random_state
            }
        }
        
        # Calculate summary statistics
        all_results = {}
        
        # Process classification results
        for dataset_name, dataset_results in classification_results.items():
            for model_name, model_results in dataset_results.items():
                key = f"classification_{dataset_name}_{model_name}"
                all_results[key] = model_results
        
        # Process regression results
        for dataset_name, dataset_results in regression_results.items():
            for model_name, model_results in dataset_results.items():
                key = f"regression_{dataset_name}_{model_name}"
                all_results[key] = model_results
        
        # Calculate success rates and average performance
        success_counts = {"autoforge": 0, "autosklearn": 0, "h2o": 0}
        total_counts = {"autoforge": 0, "autosklearn": 0, "h2o": 0}
        
        for key, result in all_results.items():
            model_name = result["model_name"].lower().replace("-", "").replace(" ", "")
            if "autoforge" in model_name:
                total_counts["autoforge"] += 1
                if result["status"] == "success":
                    success_counts["autoforge"] += 1
            elif "autosklearn" in model_name:
                total_counts["autosklearn"] += 1
                if result["status"] == "success":
                    success_counts["autosklearn"] += 1
            elif "h2o" in model_name:
                total_counts["h2o"] += 1
                if result["status"] == "success":
                    success_counts["h2o"] += 1
        
        # Calculate success rates
        success_rates = {}
        for model in ["autoforge", "autosklearn", "h2o"]:
            if total_counts[model] > 0:
                success_rates[model] = success_counts[model] / total_counts[model]
            else:
                success_rates[model] = 0.0
        
        report["summary"] = {
            "success_rates": success_rates,
            "total_benchmarks": sum(total_counts.values()),
            "available_systems": {
                "autoforge": True,
                "autosklearn": AUTOSKLEARN_AVAILABLE,
                "h2o": H2O_AVAILABLE
            }
        }
        
        return report
    
    def save_benchmark_report(self, report: Dict[str, Any], filepath: str = "benchmark_report.json"):
        """
        Save benchmark report to file
        
        Args:
            report: Benchmark report
            filepath: Output file path
        """
        import json
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Benchmark report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save benchmark report: {e}")


def run_complete_benchmark(**kwargs) -> Dict[str, Any]:
    """
    Run complete benchmark suite
    
    Args:
        **kwargs: Arguments for BenchmarkSystem
        
    Returns:
        Complete benchmark report
    """
    benchmark = BenchmarkSystem(**kwargs)
    
    # Run classification benchmarks
    classification_results = benchmark.run_classification_benchmark()
    
    # Run regression benchmarks
    regression_results = benchmark.run_regression_benchmark()
    
    # Generate report
    report = benchmark.generate_benchmark_report(classification_results, regression_results)
    
    # Save report
    benchmark.save_benchmark_report(report)
    
    return report
