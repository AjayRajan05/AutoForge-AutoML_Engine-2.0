"""
📊 AutoForge Benchmarking Integration
Integration with existing benchmarking modules
"""

try:
    from ...benchmarking.automl_benchmark import AutoMLBenchmark
    from ...benchmarking.benchmark_system import BenchmarkSystem
    from ...benchmarking.enhanced_benchmarking import EnhancedBenchmarking
    
    __all__ = ['AutoMLBenchmark', 'BenchmarkSystem', 'EnhancedBenchmarking']
    
except ImportError as e:
    # Fallback if benchmarking modules have import issues
    __all__ = []
    AutoMLBenchmark = None
    BenchmarkSystem = None
    EnhancedBenchmarking = None

# Import integration layer
from .benchmark_integration import BenchmarkingIntegrator, benchmarking_integrator, benchmark_autoforge, compare_with_existing_systems

__all__.extend(['BenchmarkingIntegrator', 'benchmarking_integrator', 'benchmark_autoforge', 'compare_with_existing_systems'])
