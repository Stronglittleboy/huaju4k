"""
Core data models for the video enhancement system.

This module defines the primary data structures used throughout the video enhancement
pipeline, including video metadata, theater-specific analysis, and processing configuration.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class LightingType(Enum):
    """Types of lighting in theater videos."""
    STAGE = "stage"
    NATURAL = "natural"
    MIXED = "mixed"


class SceneComplexity(Enum):
    """Complexity levels for theater scenes."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class StageDepth(Enum):
    """Stage depth categories."""
    SHALLOW = "shallow"
    MEDIUM = "medium"
    DEEP = "deep"


class ProcessingPriority(Enum):
    """Processing priority modes."""
    QUALITY = "quality"
    SPEED = "speed"
    BALANCED = "balanced"


class ProcessingTool(Enum):
    """Available processing tools."""
    WAIFU2X_GUI = "waifu2x-gui"
    VIDEO2X = "video2x"
    REAL_ESRGAN = "real-esrgan"


class ProcessingEnvironment(Enum):
    """Processing environments."""
    WINDOWS = "windows"
    WSL = "wsl"


@dataclass
class AudioTrack:
    """Audio track information."""
    codec: str
    bitrate: int
    channels: int
    sample_rate: int
    language: Optional[str] = None


@dataclass
class TheaterAnalysis:
    """Analysis results for theater-specific content characteristics."""
    lighting_type: LightingType
    scene_complexity: SceneComplexity
    actor_count: int
    stage_depth: StageDepth
    recommended_model: str
    processing_priority: ProcessingPriority


@dataclass
class VideoMetadata:
    """Complete metadata for video files."""
    resolution: Tuple[int, int]
    frame_rate: float
    duration: float
    codec: str
    bitrate: int
    audio_tracks: List[AudioTrack]
    theater_characteristics: Optional[TheaterAnalysis] = None
    
    @property
    def width(self) -> int:
        """Get video width."""
        return self.resolution[0]
    
    @property
    def height(self) -> int:
        """Get video height."""
        return self.resolution[1]
    
    @property
    def is_4k(self) -> bool:
        """Check if video is already 4K resolution."""
        return self.width >= 3840 and self.height >= 2160


@dataclass
class ProcessingConfig:
    """Configuration for video processing pipeline."""
    tool: ProcessingTool
    model: str
    scale_factor: int
    tile_size: int
    gpu_acceleration: bool
    denoise_level: int
    environment: ProcessingEnvironment
    color_correction: bool = True  # Enable theater-specific color correction
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.scale_factor not in [2, 3, 4]:
            raise ValueError("Scale factor must be 2, 3, or 4")
        
        if self.tile_size <= 0:
            raise ValueError("Tile size must be positive")
        
        if not 0 <= self.denoise_level <= 3:
            raise ValueError("Denoise level must be between 0 and 3")