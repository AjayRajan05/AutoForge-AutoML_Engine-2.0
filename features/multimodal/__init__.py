"""
AutoForge Multimodal Module
"""

try:
    from .intelligent_multimodal import AdvancedMultimodalAutoML
    __all__ = ['AdvancedMultimodalAutoML']
except ImportError:
    __all__ = []
    AdvancedMultimodalAutoML = None
