"""
Benchmarking integration — lightweight wrapper around classical suite.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from .classical_suite import evaluate_exit_criteria, run_classical_suite, write_benchmarks_md
    CLASSICAL_SUITE_AVAILABLE = True
except ImportError:
    CLASSICAL_SUITE_AVAILABLE = False
    run_classical_suite = None
    write_benchmarks_md = None
    evaluate_exit_criteria = None


class BenchmarkingIntegrator:
    """Integration layer for benchmarking."""

    def __init__(self):
        self.available_benchmarkers = {
            "classical_suite": CLASSICAL_SUITE_AVAILABLE,
        }
        self.benchmark_history: List[Dict[str, Any]] = []

    def benchmark_autoforge(
        self,
        autoforge_instance,
        test_datasets: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Benchmark AutoForge on a list of {name, X, y} datasets."""
        try:
            results: Dict[str, Any] = {
                "autoforge_results": {},
                "summary": {},
                "timestamp": time.time(),
                "datasets_tested": len(test_datasets),
            }
            total_time = 0.0
            all_scores: List[float] = []

            for i, dataset in enumerate(test_datasets):
                dataset_name = dataset.get("name", f"dataset_{i}")
                X = dataset["X"]
                y = dataset["y"]
                start_time = time.time()
                autoforge_instance.fit(X, y)
                fit_time = time.time() - start_time
                score = autoforge_instance.best_score
                predictions = autoforge_instance.predict(X)

                if len(np.unique(y)) < 20:
                    from sklearn.metrics import accuracy_score, f1_score
                    metrics = {
                        "accuracy": accuracy_score(y, predictions),
                        "f1_score": f1_score(y, predictions, average="weighted"),
                    }
                else:
                    from sklearn.metrics import r2_score, mean_squared_error
                    metrics = {
                        "r2_score": r2_score(y, predictions),
                        "mse": mean_squared_error(y, predictions),
                    }

                results["autoforge_results"][dataset_name] = {
                    "score": score,
                    "fit_time": fit_time,
                    "metrics": metrics,
                    "model_type": type(autoforge_instance.best_model).__name__,
                    "n_features": len(
                        autoforge_instance.training_metadata.get("selected_features", [])
                    ),
                    "dataset_shape": getattr(X, "shape", None),
                    "status": "success",
                }
                all_scores.append(float(score or 0.0))
                total_time += fit_time

            results["summary"] = {
                "average_score": float(np.mean(all_scores)) if all_scores else 0.0,
                "best_score": float(np.max(all_scores)) if all_scores else 0.0,
                "worst_score": float(np.min(all_scores)) if all_scores else 0.0,
                "std_score": float(np.std(all_scores)) if all_scores else 0.0,
                "total_time": total_time,
                "average_time": total_time / len(test_datasets) if test_datasets else 0.0,
                "success_rate": 1.0,
            }
            self.benchmark_history.append(results)
            return results
        except Exception as exc:
            logger.error("AutoForge benchmarking failed: %s", exc)
            return {"error": str(exc)}

    def compare_with_existing_systems(
        self, autoforge_instance, test_datasets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run AutoForge benchmark; classical suite comparison is via ``run_classical_suite``."""
        autoforge_results = self.benchmark_autoforge(autoforge_instance, test_datasets)
        return {
            "autoforge": autoforge_results,
            "existing_systems": {},
            "comparison": {},
            "timestamp": time.time(),
        }

    def run_classical_suite(self, **kwargs) -> List[Dict[str, Any]]:
        if not CLASSICAL_SUITE_AVAILABLE:
            raise ImportError("benchmarking.classical_suite is not available")
        return run_classical_suite(**kwargs)

    def get_benchmark_history(self) -> List[Dict[str, Any]]:
        return self.benchmark_history


benchmarking_integrator = BenchmarkingIntegrator()


def benchmark_autoforge(
    autoforge_instance,
    test_datasets: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return benchmarking_integrator.benchmark_autoforge(autoforge_instance, test_datasets, config)


def compare_with_existing_systems(
    autoforge_instance, test_datasets: List[Dict[str, Any]]
) -> Dict[str, Any]:
    return benchmarking_integrator.compare_with_existing_systems(autoforge_instance, test_datasets)
