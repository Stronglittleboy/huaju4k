"""
Core Data Models

Data structures used throughout the video enhancement pipeline.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    input_path: str
    output_path: str
    target_resolution: Tuple[int, int]
    ai_model: str
    tile_size: int
    batch_size: int
    gpu_acceleration: bool
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.tile_size <= 0:
            raise ValueError("Tile size must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")


@dataclass
class AudioConfig:
    """Configuration for audio optimization."""
    noise_reduction_strength: float = 0.3  # 0.0-1.0
    dialogue_enhancement: float = 0.4      # 0.0-1.0
    dynamic_range_target: float = -23.0    # dB
    preserve_naturalness: bool = True
    theater_preset: str = "medium"         # "small", "medium", "large"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.noise_reduction_strength <= 1.0:
            raise ValueError("Noise reduction strength must be between 0.0 and 1.0")
        if not 0.0 <= self.dialogue_enhancement <= 1.0:
            raise ValueError("Dialogue enhancement must be between 0.0 and 1.0")
        if self.theater_preset not in ["small", "medium", "large"]:
            raise ValueError("Theater preset must be 'small', 'medium', or 'large'")


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    cpu_threads: int
    gpu_memory_limit: float  # GB
    memory_optimization: bool
    parallel_processing: bool
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.cpu_threads <= 0:
            raise ValueError("CPU threads must be positive")
        if self.gpu_memory_limit <= 0:
            raise ValueError("GPU memory limit must be positive")


@dataclass
class AudioAnalysis:
    """Results of theater audio analysis."""
    noise_profile: np.ndarray
    dialogue_frequency_range: Tuple[float, float]
    dynamic_range: float
    peak_levels: List[float]
    recommended_settings: AudioConfig


@dataclass
class QualityMetrics:
    """Audio quality validation metrics."""
    snr_improvement: float
    thd_level: float
    dialogue_clarity_score: float
    naturalness_score: float
    distortion_detected: bool