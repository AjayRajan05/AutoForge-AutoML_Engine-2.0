"""
AutoForge NAS Module
"""

try:
    from .revolutionary_nas import AdvancedNAS
    from .search import NeuralArchitectureSearch
    __all__ = ['AdvancedNAS', 'NeuralArchitectureSearch']
except ImportError:
    __all__ = []
    AdvancedNAS = None
    NeuralArchitectureSearch = None
