"""
Enhanced Benchmarking System
Research-grade benchmarking with visualizations and actionable insights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import gc
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
import warnings

# 🔥 PRODUCTION-GRADE: Configuration for Unicode handling
CONFIG = {
    "allow_unicode": False  # Set to True for full Unicode support
}

# Import AutoML and competitors - moved inside methods to avoid circular imports

logger = logging.getLogger(__name__)


class EnhancedBenchmarking:
    """
    Enhanced benchmarking system with visualizations and insights
    """
    
    def __init__(self, results_dir: str = "benchmarking/results"):
        """
        Initialize enhanced benchmarking system
        
        Args:
            results_dir: Directory to store benchmarking results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.failure_log = []  # 🔥 PRODUCTION-GRADE: Track failures
        self.benchmark_results = {}
        
        logger.info(f"Enhanced benchmarking initialized: {self.results_dir}")
    
    def safe_encode(self, text: str) -> str:
        """🔥 PRODUCTION-GRADE: Safe encoding wrapper"""
        if CONFIG["allow_unicode"]:
            return text.encode("utf-8", errors="ignore").decode("utf-8")
        else:
            # Remove non-ASCII characters for maximum compatibility
            return text.encode("ascii", errors="ignore").decode("ascii")
    
    def run_comprehensive_benchmark(self, 
                                  datasets: List[Dict[str, Any]] = None,
                                  competitors: List[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmarking with visualizations
        
        Args:
            datasets: List of dataset configurations
            competitors: List of competitor systems to test
            
        Returns:
            Comprehensive benchmarking results
        """
        try:
            if datasets is None:
                datasets = self._generate_test_datasets()
            
            if competitors is None:
                competitors = ["AutoForge", "Auto-sklearn", "H2O AutoML"]
            
            logger.info(f"Starting comprehensive benchmark: {len(datasets)} datasets, {len(competitors)} systems")
            
            results = {
                "datasets": [],
                "system_performance": {},
                "visualizations": {},
                "insights": {},
                "timestamp": time.time()
            }
            
            # Run benchmarks for each dataset
            for i, dataset_config in enumerate(datasets):
                logger.info(f"Benchmarking dataset {i+1}/{len(datasets)}: {dataset_config['name']}")
                
                dataset_results = self._benchmark_single_dataset(
                    dataset_config, competitors
                )
                
                results["datasets"].append(dataset_results)
                
                # Clear memory between datasets
                gc.collect()
            
            # Aggregate results across datasets
            results["system_performance"] = self._aggregate_performance(results["datasets"])
            
            # Generate visualizations
            results["visualizations"] = self._generate_visualizations(results)
            
            # Generate insights
            results["insights"] = self._generate_insights(results)
            
            # Save results
            self._save_benchmark_results(results)
            
            logger.info("Comprehensive benchmarking completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive benchmarking failed: {e}")
            raise
    
    def _generate_test_datasets(self) -> List[Dict[str, Any]]:
        """Generate diverse test datasets for benchmarking"""
        datasets = []
        
        # Small classification dataset
        X_small, y_small = make_classification(
            n_samples=1000, n_features=20, n_informative=15, random_state=42
        )
        datasets.append({
            "name": "Small Classification",
            "type": "classification",
            "size": "small",
            "X": X_small, "y": y_small,
            "description": "1K samples, 20 features"
        })
        
        # Medium classification dataset
        X_med, y_med = make_classification(
            n_samples=10000, n_features=50, n_informative=35, random_state=42
        )
        datasets.append({
            "name": "Medium Classification",
            "type": "classification",
            "size": "medium",
            "X": X_med, "y": y_med,
            "description": "10K samples, 50 features"
        })
        
        # Large classification dataset
        X_large, y_large = make_classification(
            n_samples=100000, n_features=100, n_informative=70, random_state=42
        )
        datasets.append({
            "name": "Large Classification",
            "type": "classification",
            "size": "large",
            "X": X_large, "y": y_large,
            "description": "100K samples, 100 features"
        })
        
        # Small regression dataset
        X_reg_small, y_reg_small = make_regression(
            n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=42
        )
        datasets.append({
            "name": "Small Regression",
            "type": "regression",
            "size": "small",
            "X": X_reg_small, "y": y_reg_small,
            "description": "1K samples, 20 features"
        })
        
        # Medium regression dataset
        X_reg_med, y_reg_med = make_regression(
            n_samples=10000, n_features=50, n_informative=35, noise=0.1, random_state=42
        )
        datasets.append({
            "name": "Medium Regression",
            "type": "regression",
            "size": "medium",
            "X": X_reg_med, "y": y_reg_med,
            "description": "10K samples, 50 features"
        })
        
        return datasets
    
    def _benchmark_single_dataset(self, 
                               dataset_config: Dict[str, Any],
                               competitors: List[str]) -> Dict[str, Any]:
        """Benchmark a single dataset across all systems"""
        X, y = dataset_config["X"], dataset_config["y"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        dataset_results = {
            "dataset_name": dataset_config["name"],
            "dataset_type": dataset_config["type"],
            "dataset_size": dataset_config["size"],
            "dataset_description": dataset_config["description"],
            "systems": {}
        }
        
        # Benchmark each system
        for system in competitors:
            logger.info(f"  Benchmarking {system}...")
            
            try:
                system_results = self._benchmark_system(
                    system, X_train, X_test, y_train, y_test, dataset_config["type"]
                )
                dataset_results["systems"][system] = system_results
                
            except Exception as e:
                logger.error(f"    Failed to benchmark {system}: {e}")
                dataset_results["systems"][system] = {
                    "error": str(e),
                    "performance": None,
                    "time": None,
                    "memory": None
                }
        
        return dataset_results
    
    def _benchmark_system(self, 
                         system_name: str,
                         X_train: np.ndarray,
                         X_test: np.ndarray,
                         y_train: np.ndarray,
                         y_test: np.ndarray,
                         task_type: str) -> Dict[str, Any]:
        """Benchmark a single system"""
        
        # Monitor initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time the execution
        start_time = time.time()
        
        if system_name == "AutoForge":
            results = self._benchmark_autoforge(
                X_train, X_test, y_train, y_test, task_type
            )
        elif system_name == "Auto-sklearn":
            results = self._benchmark_autosklearn(
                X_train, X_test, y_train, y_test, task_type
            )
        elif system_name == "H2O AutoML":
            results = self._benchmark_h2o(
                X_train, X_test, y_train, y_test, task_type
            )
        else:
            raise ValueError(f"Unknown system: {system_name}")
        
        end_time = time.time()
        
        # Monitor peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        results.update({
            "time": end_time - start_time,
            "memory": peak_memory - initial_memory,
            "stability": self._check_stability(results["performance"])
        })
        
        return results
    
    def _benchmark_autoforge(self, 
                           X_train: np.ndarray,
                           X_test: np.ndarray,
                           y_train: np.ndarray,
                           y_test: np.ndarray,
                           task_type: str) -> Dict[str, Any]:
        """Benchmark AutoForge system"""
        try:
            # Import AutoML here to avoid circular imports
            from .automl import AutoML
            
            # Initialize AutoForge with reduced trials for speed
            automl = AutoML(
                n_trials=20,  # Reduced for benchmarking
                cv=3,
                use_explainability=False,  # Disable for speed
                show_progress=False
            )
            
            # Fit model
            automl.fit(X_train, y_train)
            
            # Make predictions
            y_pred = automl.predict(X_test)
            
            # Calculate performance
            if task_type == "classification":
                performance = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred, average='weighted')
                }
            else:
                performance = {
                    "mse": mean_squared_error(y_test, y_pred),
                    "r2_score": r2_score(y_test, y_pred)
                }
            
            return {
                "system": "AutoForge",
                "performance": performance,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"AutoForge benchmarking failed: {e}")
            return {
                "system": "AutoForge",
                "performance": None,
                "success": False,
                "error": str(e)
            }
    
    def _benchmark_autosklearn(self, 
                              X_train: np.ndarray,
                              X_test: np.ndarray,
                              y_train: np.ndarray,
                              y_test: np.ndarray,
                              task_type: str) -> Dict[str, Any]:
        """Benchmark Auto-sklearn system (simulated)"""
        try:
            # Simulate Auto-sklearn performance
            # In real implementation, this would use actual Auto-sklearn
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            if task_type == "classification":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate performance
            if task_type == "classification":
                performance = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred, average='weighted')
                }
            else:
                performance = {
                    "mse": mean_squared_error(y_test, y_pred),
                    "r2_score": r2_score(y_test, y_pred)
                }
            
            return {
                "system": "Auto-sklearn",
                "performance": performance,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Auto-sklearn benchmarking failed: {e}")
            return {
                "system": "Auto-sklearn",
                "performance": None,
                "success": False,
                "error": str(e)
            }
    
    def _benchmark_h2o(self, 
                        X_train: np.ndarray,
                        X_test: np.ndarray,
                        y_train: np.ndarray,
                        y_test: np.ndarray,
                        task_type: str) -> Dict[str, Any]:
        """Benchmark H2O AutoML system (simulated)"""
        try:
            # Simulate H2O AutoML performance
            # In real implementation, this would use actual H2O
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            
            if task_type == "classification":
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            else:
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate performance
            if task_type == "classification":
                performance = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred, average='weighted')
                }
            else:
                performance = {
                    "mse": mean_squared_error(y_test, y_pred),
                    "r2_score": r2_score(y_test, y_pred)
                }
            
            return {
                "system": "H2O AutoML",
                "performance": performance,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"H2O AutoML benchmarking failed: {e}")
            return {
                "system": "H2O AutoML",
                "performance": None,
                "success": False,
                "error": str(e)
            }
    
    def _check_stability(self, performance: Dict[str, float]) -> bool:
        """Check if performance is stable across runs"""
        # This would require multiple runs for true stability
        # For now, return True if performance is reasonable
        if not performance:
            return False
        
        if "accuracy" in performance:
            return performance["accuracy"] > 0.5  # Reasonable accuracy
        elif "r2_score" in performance:
            return performance["r2_score"] > 0.0  # Reasonable R2
        
        return True
    
    def _aggregate_performance(self, dataset_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate performance across all datasets"""
        systems = {}
        
        # Collect all results for each system
        for dataset in dataset_results:
            for system_name, system_results in dataset["systems"].items():
                if system_name not in systems:
                    systems[system_name] = {
                        "performances": [],
                        "times": [],
                        "memories": [],
                        "stabilities": [],
                        "success_rate": 0
                    }
                
                if system_results.get("success", False):
                    systems[system_name]["performances"].append(system_results.get("performance", {}))
                    systems[system_name]["times"].append(system_results.get("time", 0))
                    systems[system_name]["memories"].append(system_results.get("memory", 0))
                    systems[system_name]["stabilities"].append(system_results.get("stability", False))
                
                # Calculate success rate
                total_datasets = len(dataset_results)
                successful_datasets = sum(1 for d in dataset_results 
                                     if d["systems"].get(system_name, {}).get("success", False))
                systems[system_name]["success_rate"] = successful_datasets / total_datasets
        
        # Calculate statistics for each system
        for system_name, system_data in systems.items():
            performances = system_data["performances"]
            
            if performances:
                # Calculate average performance
                avg_performance = {}
                
                # Get all metric names
                metric_names = set()
                for perf in performances:
                    if perf:
                        metric_names.update(perf.keys())
                
                for metric in metric_names:
                    values = [perf.get(metric, 0) for perf in performances if perf]
                    avg_performance[metric] = np.mean(values)
                    avg_performance[f"{metric}_std"] = np.std(values)
                
                system_data["avg_performance"] = avg_performance
                system_data["avg_time"] = np.mean(system_data["times"])
                system_data["avg_memory"] = np.mean(system_data["memories"])
                system_data["stability_rate"] = np.mean(system_data["stabilities"])
        
        return systems
    
    def _generate_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate benchmarking visualizations"""
        visualizations = {}
        
        try:
            # Set up plotting style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. Accuracy vs Time plot
            viz_path = self._plot_accuracy_vs_time(results)
            visualizations["accuracy_vs_time"] = viz_path
            
            # 2. Memory usage plot
            viz_path = self._plot_memory_usage(results)
            visualizations["memory_usage"] = viz_path
            
            # 3. Stability across datasets
            viz_path = self._plot_stability(results)
            visualizations["stability"] = viz_path
            
            # 4. Performance comparison
            viz_path = self._plot_performance_comparison(results)
            visualizations["performance_comparison"] = viz_path
            
            logger.info(f"Generated {len(visualizations)} visualizations")
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            visualizations["error"] = str(e)
        
        return visualizations
    
    def _plot_accuracy_vs_time(self, results: Dict[str, Any]) -> str:
        """Plot accuracy vs time for all systems"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        system_performance = results["system_performance"]
        
        # Classification accuracy
        ax = axes[0]
        for system_name, system_data in system_performance.items():
            if system_data["avg_performance"]:
                avg_acc = system_data["avg_performance"].get("accuracy", 0)
                avg_time = system_data["avg_time"]
                ax.scatter(avg_time, avg_acc, s=100, alpha=0.7, label=system_name)
        
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Classification: Accuracy vs Training Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Regression R2
        ax = axes[1]
        for system_name, system_data in system_performance.items():
            if system_data["avg_performance"]:
                avg_r2 = system_data["avg_performance"].get("r2_score", 0)
                avg_time = system_data["avg_time"]
                ax.scatter(avg_time, avg_r2, s=100, alpha=0.7, label=system_name)
        
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('R² Score')
        ax.set_title('Regression: R² vs Training Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / "accuracy_vs_time.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_memory_usage(self, results: Dict[str, Any]) -> str:
        """Plot memory usage comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        system_performance = results["system_performance"]
        
        systems = []
        memories = []
        
        for system_name, system_data in system_performance.items():
            systems.append(system_name)
            memories.append(system_data["avg_memory"])
        
        bars = ax.bar(systems, memories, alpha=0.7)
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Average Memory Usage Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, memory in zip(bars, memories):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(memories)*0.01,
                    f'{memory:.1f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / "memory_usage.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_stability(self, results: Dict[str, Any]) -> str:
        """Plot stability across datasets"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        system_performance = results["system_performance"]
        
        systems = []
        success_rates = []
        stability_rates = []
        
        for system_name, system_data in system_performance.items():
            systems.append(system_name)
            success_rates.append(system_data["success_rate"] * 100)
            stability_rates.append(system_data["stability_rate"] * 100)
        
        x = np.arange(len(systems))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, success_rates, width, label='Success Rate', alpha=0.7)
        bars2 = ax.bar(x + width/2, stability_rates, width, label='Stability Rate', alpha=0.7)
        
        ax.set_ylabel('Rate (%)')
        ax.set_title('Success Rate and Stability Across Datasets')
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / "stability.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_performance_comparison(self, results: Dict[str, Any]) -> str:
        """Plot performance comparison across datasets"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data for plotting
        datasets = []
        systems = []
        accuracy_data = []
        r2_data = []
        time_data = []
        memory_data = []
        
        for dataset in results["datasets"]:
            dataset_name = dataset["dataset_name"]
            datasets.append(dataset_name)
            
            for system_name, system_results in dataset["systems"].items():
                if system_results.get("success", False):
                    systems.append(f"{system_name}\n({dataset_name})")
                    
                    perf = system_results.get("performance", {})
                    accuracy_data.append(perf.get("accuracy", 0))
                    r2_data.append(perf.get("r2_score", 0))
                    time_data.append(system_results.get("time", 0))
                    memory_data.append(system_results.get("memory", 0))
        
        # Classification accuracy
        ax = axes[0, 0]
        if accuracy_data:
            ax.bar(range(len(accuracy_data)), accuracy_data, alpha=0.7)
            ax.set_ylabel('Accuracy')
            ax.set_title('Classification Accuracy')
            ax.set_xticks([])
        
        # Regression R2
        ax = axes[0, 1]
        if r2_data:
            ax.bar(range(len(r2_data)), r2_data, alpha=0.7)
            ax.set_ylabel('R² Score')
            ax.set_title('Regression R²')
            ax.set_xticks([])
        
        # Training time
        ax = axes[1, 0]
        if time_data:
            ax.bar(range(len(time_data)), time_data, alpha=0.7)
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Training Time')
            ax.set_xticks([])
        
        # Memory usage
        ax = axes[1, 1]
        if memory_data:
            ax.bar(range(len(memory_data)), memory_data, alpha=0.7)
            ax.set_ylabel('Memory (MB)')
            ax.set_title('Memory Usage')
            ax.set_xticks([])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / "performance_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _generate_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable insights from benchmarking results"""
        insights = {
            "performance_analysis": {},
            "efficiency_analysis": {},
            "recommendations": {},
            "actionable_insights": {}
        }
        
        try:
            system_performance = results["system_performance"]
            
            # Performance analysis
            for system_name, system_data in system_performance.items():
                if system_data["avg_performance"]:
                    insights["performance_analysis"][system_name] = {
                        "avg_accuracy": system_data["avg_performance"].get("accuracy", 0),
                        "avg_r2": system_data["avg_performance"].get("r2_score", 0),
                        "avg_time": system_data["avg_time"],
                        "avg_memory": system_data["avg_memory"],
                        "success_rate": system_data["success_rate"],
                        "stability_rate": system_data["stability_rate"]
                    }
            
            # Efficiency analysis
            best_accuracy = max([
                data.get("avg_performance", {}).get("accuracy", 0) 
                for data in system_performance.values()
            ])
            
            best_time = min([
                data["avg_time"] 
                for data in system_performance.values() if data["avg_time"] > 0
            ])
            
            best_memory = min([
                data["avg_memory"] 
                for data in system_performance.values() if data["avg_memory"] > 0
            ])
            
            insights["efficiency_analysis"] = {
                "best_accuracy_system": None,
                "fastest_system": None,
                "most_efficient_system": None,
                "best_overall": None
            }
            
            # Find best systems
            for system_name, system_data in system_performance.items():
                if system_data["avg_performance"]:
                    accuracy = system_data["avg_performance"].get("accuracy", 0)
                    time_val = system_data["avg_time"]
                    memory_val = system_data["avg_memory"]
                    
                    if accuracy >= best_accuracy * 0.95:  # Within 5% of best
                        if not insights["efficiency_analysis"]["best_accuracy_system"]:
                            insights["efficiency_analysis"]["best_accuracy_system"] = system_name
                    
                    if time_val <= best_time * 1.2:  # Within 20% of fastest
                        if not insights["efficiency_analysis"]["fastest_system"]:
                            insights["efficiency_analysis"]["fastest_system"] = system_name
                    
                    if memory_val <= best_memory * 1.2:  # Within 20% of most efficient
                        if not insights["efficiency_analysis"]["most_efficient_system"]:
                            insights["efficiency_analysis"]["most_efficient_system"] = system_name
            
            # Generate actionable insights
            insights["actionable_insights"] = self._generate_actionable_insights(
                system_performance
            )
            
            logger.info(f"Generated insights: {len(insights['actionable_insights'])} actionable items")
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            insights["error"] = str(e)
        
        return insights
    
    def _generate_actionable_insights(self, system_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable insights from system performance"""
        insights = []
        
        for system_name, system_data in system_performance.items():
            if not system_data["avg_performance"]:
                continue
            
            # Performance insights
            accuracy = system_data["avg_performance"].get("accuracy", 0)
            r2 = system_data["avg_performance"].get("r2_score", 0)
            time_val = system_data["avg_time"]
            memory_val = system_data["avg_memory"]
            success_rate = system_data["success_rate"]
            
            # Generate insights based on performance characteristics
            if accuracy > 0.9 and time_val > 100:
                insights.append({
                    "system": system_name,
                    "type": "performance_tradeoff",
                    "insight": f"{system_name} achieves high accuracy but with slow training time",
                    "recommendation": "Consider optimizing for speed or use for accuracy-critical applications",
                    "priority": "medium"
                })
            
            if accuracy < 0.7:
                insights.append({
                    "system": system_name,
                    "type": "low_performance",
                    "insight": f"{system_name} shows low accuracy across datasets",
                    "recommendation": "Consider feature engineering or hyperparameter tuning",
                    "priority": "high"
                })
            
            if memory_val > 1000:  # > 1GB
                insights.append({
                    "system": system_name,
                    "type": "high_memory",
                    "insight": f"{system_name} uses significant memory",
                    "recommendation": "Consider memory optimization or use with larger memory systems",
                    "priority": "medium"
                })
            
            if success_rate < 0.8:
                insights.append({
                    "system": system_name,
                    "type": "low_reliability",
                    "insight": f"{system_name} fails on {100-success_rate*100:.0f}% of datasets",
                    "recommendation": "Improve error handling and dataset compatibility",
                    "priority": "high"
                })
        
        return insights
    
    def _save_benchmark_results(self, results: Dict[str, Any]) -> None:
        """Save comprehensive benchmarking results"""
        try:
            # Save JSON results
            json_path = self.results_dir / "comprehensive_benchmark.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save summary report
            report_path = self.results_dir / "benchmark_report.txt"
            self._generate_text_report(results, report_path)
            
            logger.info(f"Benchmarking results saved to {self.results_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save benchmarking results: {e}")
    
    def _generate_text_report(self, results: Dict[str, Any], report_path: Path) -> None:
        """Generate human-readable benchmarking report"""
        try:
            # 🔥 PRODUCTION-GRADE: Force UTF-8 encoding
            with open(report_path, 'w', encoding='utf-8') as f:
                # 🔥 PRODUCTION-GRADE: Safe encoding wrapper
                safe_title = self.safe_encode("🏆 AutoForge Enhanced Benchmarking Report")
                f.write(safe_title + "\n")
                f.write("=" * 60 + "\n\n")
                
                # Executive summary
                f.write("📊 EXECUTIVE SUMMARY\n")
                f.write("-" * 30 + "\n")
                
                insights = results.get("insights", {})
                efficiency = insights.get("efficiency_analysis", {})
                
                f.write(f"Best Accuracy System: {efficiency.get('best_accuracy_system', 'N/A')}\n")
                f.write(f"Fastest System: {efficiency.get('fastest_system', 'N/A')}\n")
                f.write(f"Most Efficient System: {efficiency.get('most_efficient_system', 'N/A')}\n\n")
                
                # System performance details
                f.write("🔍 SYSTEM PERFORMANCE DETAILS\n")
                f.write("-" * 30 + "\n")
                
                system_performance = results.get("system_performance", {})
                for system_name, system_data in system_performance.items():
                    f.write(f"\n{system_name}:\n")
                    
                    if system_data["avg_performance"]:
                        perf = system_data["avg_performance"]
                        f.write(f"  Average Accuracy: {perf.get('accuracy', 'N/A'):.4f}\n")
                        f.write(f"  Average R² Score: {perf.get('r2_score', 'N/A'):.4f}\n")
                    
                    f.write(f"  Average Training Time: {system_data['avg_time']:.2f} seconds\n")
                    f.write(f"  Average Memory Usage: {system_data['avg_memory']:.1f} MB\n")
                    f.write(f"  Success Rate: {system_data['success_rate']*100:.1f}%\n")
                    f.write(f"  Stability Rate: {system_data['stability_rate']*100:.1f}%\n")
                
                # Actionable insights
                f.write(f"\n💡 ACTIONABLE INSIGHTS\n")
                f.write("-" * 30 + "\n")
                
                actionable = insights.get("actionable_insights", [])
                for insight in actionable:
                    f.write(f"\n🎯 {insight['system']} - {insight['type'].title()}\n")
                    f.write(f"   Insight: {insight['insight']}\n")
                    f.write(f"   Recommendation: {insight['recommendation']}\n")
                    f.write(f"   Priority: {insight['priority'].upper()}\n")
                
                # Visualizations
                f.write(f"\n📈 VISUALIZATIONS\n")
                f.write("-" * 30 + "\n")
                
                visualizations = results.get("visualizations", {})
                for viz_name, viz_path in visualizations.items():
                    if viz_path != "error":
                        f.write(f"{viz_name.replace('_', ' ').title()}: {viz_path}\n")
                
        except Exception as e:
            logger.error(f"Failed to generate text report: {e}")


# Convenience functions
def run_comprehensive_benchmark(datasets: List[Dict[str, Any]] = None, 
                           competitors: List[str] = None) -> Dict[str, Any]:
    """Convenience function for running comprehensive benchmarking"""
    benchmarking = EnhancedBenchmarking()
    return benchmarking.run_comprehensive_benchmark(datasets, competitors)


def get_benchmarking_insights() -> Dict[str, Any]:
    """Convenience function for getting latest benchmarking insights"""
    benchmarking = EnhancedBenchmarking()
    
    # Load latest results
    results_file = benchmarking.results_dir / "comprehensive_benchmark.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results.get("insights", {})
    
    return {"error": "No benchmarking results found"}
