"""
📊 AutoForge Tracking Integration
Integration with existing tracking modules
"""

try:
    from ...tracking.experiment import Experiment
    from ...tracking.logger import ExperimentLogger
    from ...tracking.storage import ExperimentStorage
    
    __all__ = ['Experiment', 'ExperimentLogger', 'ExperimentStorage']
    
except ImportError as e:
    # Fallback if tracking modules have import issues
    __all__ = []
    Experiment = None
    ExperimentLogger = None
    ExperimentStorage = None

# Import integration layer
from .tracking_integration import TrackingIntegrator, tracking_integrator, start_autoforge_experiment, log_autoforge_training, end_autoforge_experiment

__all__.extend(['TrackingIntegrator', 'tracking_integrator', 'start_autoforge_experiment', 'log_autoforge_training', 'end_autoforge_experiment'])
