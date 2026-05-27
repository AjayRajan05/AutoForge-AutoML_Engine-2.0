"""
📥 AutoForge Input/Output Module
Standardized data structures and validation
"""

from .input_types import (
    AutoMLInput, 
    AutoMLOutput, 
    ValidationResult,
    get_test_input,
    get_small_test_input,
    get_large_test_input,
    get_text_test_input,
    get_time_series_test_input
)
from .input_validator import InputValidator

__all__ = [
    'AutoMLInput',
    'AutoMLOutput', 
    'ValidationResult',
    'InputValidator',
    'get_test_input',
    'get_small_test_input',
    'get_large_test_input',
    'get_text_test_input',
    'get_time_series_test_input'
]
