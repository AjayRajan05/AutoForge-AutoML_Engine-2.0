"""
⚡ AutoForge Optimizer Integration
Integration with existing optimizer modules
"""

try:
    from .adaptive_optimizer import AdaptiveOptimizer
    from .optuna_search import OptunaOptimizer
    from .search_space import SearchSpace
    
    __all__ = ['AdaptiveOptimizer', 'OptunaOptimizer', 'SearchSpace']
    
except ImportError as e:
    # Fallback if optimizer modules have import issues
    __all__ = []
    AdaptiveOptimizer = None
    OptunaOptimizer = None
    SearchSpace = None

# Import integration layer
from .optimizer_integration import OptimizerIntegrator, optimizer_integrator, optimize_autoforge_model, compare_autoforge_optimizers

__all__.extend(['OptimizerIntegrator', 'optimizer_integrator', 'optimize_autoforge_model', 'compare_autoforge_optimizers'])
