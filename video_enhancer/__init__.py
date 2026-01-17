"""
Video Enhancement System for Theater Drama Videos

A Python-based system for enhancing theater drama videos from 1080p/720p to 4K resolution
using free, open-source AI-powered tools.
"""

__version__ = "0.1.0"
__author__ = "Video Enhancement System"

from .models import VideoMetadata, TheaterAnalysis, ProcessingConfig
from .config import ConfigManager

__all__ = [
    "VideoMetadata",
    "TheaterAnalysis", 
    "ProcessingConfig",
    "ConfigManager"
]