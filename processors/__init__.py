"""
AutoForge Processors Integration
"""

import bootstrap  # noqa: F401

try:
    from .tabular_processor import TabularProcessor
    from .text_processor import TextProcessor
    from .image_processor import ImageProcessor
    from .time_series_processor import TimeSeriesProcessor
    __all__ = ['TabularProcessor', 'TextProcessor', 'ImageProcessor', 'TimeSeriesProcessor']
except ImportError:
    __all__ = []
    TabularProcessor = None
    TextProcessor = None
    ImageProcessor = None
    TimeSeriesProcessor = None

from .processor_integration import ProcessorIntegrator, processor_integrator, process_data_with_autoforge

__all__.extend(['ProcessorIntegrator', 'processor_integrator', 'process_data_with_autoforge'])
