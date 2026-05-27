"""
🧠 AutoForge Intelligence Module
Dataset analysis, strategy selection, and data type detection
"""

import bootstrap  # noqa: F401 — adds project root to sys.path

from .intelligence_engine import IntelligenceEngine, DatasetProfile
from .dataset_analyzer import DatasetIntelligence
from .strategy_selector import StrategySelector
from .data_type_detector import DataTypeDetector

__all__ = [
    'IntelligenceEngine', 
    'DatasetProfile',
    'DatasetIntelligence', 
    'StrategySelector',
    'DataTypeDetector'
]
