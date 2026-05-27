"""
💻 AutoForge CLI Integration
Integration with existing CLI modules
"""

try:
    from ...cli.commands import CLICommands
    from ...cli.main import main as original_main
    
    __all__ = ['CLICommands', 'original_main']
    
except ImportError as e:
    # Fallback if CLI modules have import issues
    __all__ = []
    CLICommands = None
    original_main = None

# Import integration layer
from .cli_integration import CLIIntegrator, cli_integrator, create_autoforge_cli

__all__.extend(['CLIIntegrator', 'cli_integrator', 'create_autoforge_cli'])
