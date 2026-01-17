"""
Core Processing Layer

Contains the main business logic for video and audio processing.
"""

from .interfaces import IVideoProcessor, IAudioOptimizer, IPerformanceManager
from .models import VideoConfig, AudioConfig, PerformanceConfig, AudioAnalysis, QualityMetrics
from .audio_optimizer import TheaterAudioOptimizer

__all__ = [
    "IVideoProcessor",
    "IAudioOptimizer",
    "IPerformanceManager",
    "TheaterAudioOptimizer",
    "VideoConfig",
    "AudioConfig", 
    "PerformanceConfig",
    "AudioAnalysis",
    "QualityMetrics"
]