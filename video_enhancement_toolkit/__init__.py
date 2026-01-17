"""
Video Enhancement Toolkit

A professional, modular video enhancement system for theater video processing
with optimized audio enhancement, performance maximization, and comprehensive logging.
"""

__version__ = "1.0.0"
__author__ = "Video Enhancement Toolkit"

from .core.interfaces import IVideoProcessor, IAudioOptimizer, IPerformanceManager
from .infrastructure.interfaces import ILogger, IProgressTracker, IConfigurationManager
from .cli.interfaces import ICLIController, IMenuSystem

__all__ = [
    "IVideoProcessor",
    "IAudioOptimizer", 
    "IPerformanceManager",
    "ILogger",
    "IProgressTracker",
    "IConfigurationManager",
    "ICLIController",
    "IMenuSystem"
]