"""
AutoForge Distributed Module
"""

try:
    from .intelligent_distributed import IntelligentDistributed
    __all__ = ['IntelligentDistributed']
except ImportError:
    # Fallback if modules have import issues
    __all__ = []
    IntelligentDistributed = None
