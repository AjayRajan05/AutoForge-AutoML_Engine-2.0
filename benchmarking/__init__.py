"""AutoForge benchmarking — classical suite and integration helpers."""

from .benchmark_integration import (
    BenchmarkingIntegrator,
    benchmarking_integrator,
    benchmark_autoforge,
    compare_with_existing_systems,
)
from .classical_suite import (
    evaluate_exit_criteria,
    run_classical_suite,
    write_benchmarks_md,
)

__all__ = [
    "BenchmarkingIntegrator",
    "benchmarking_integrator",
    "benchmark_autoforge",
    "compare_with_existing_systems",
    "run_classical_suite",
    "write_benchmarks_md",
    "evaluate_exit_criteria",
]
