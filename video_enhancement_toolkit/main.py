"""
Main Application Entry Point

Demonstrates the modular architecture and dependency injection setup.
"""

import sys
from typing import Optional

from .container import container
from .module_config import configure_default_modules
from .cli.interfaces import ICLIController


def main(config_path: Optional[str] = None) -> int:
    """Main application entry point.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Configure dependency injection container
        configure_default_modules()
        
        # Resolve CLI controller
        cli_controller = container.resolve(ICLIController)
        cli_controller.run()
        
        return 0
        
    except Exception as e:
        print(f"Error starting application: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())