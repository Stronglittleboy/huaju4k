"""
Core processing components for huaju4k video enhancement.
"""

from .interfaces import (
    VideoProcessor,
    AudioEnhancer,
    MemoryManager,
    ProgressTracker,
    ConfigurationManager,
    CheckpointSystem,
    ErrorHandler,
    AIModelManager as AIModelManagerInterface,
    TileProcessor as TileProcessorInterface
)

from .memory_manager import ConservativeMemoryManager
from .progress_tracker import MultiStageProgressTracker
from .ai_model_manager import AIModelManager, StrategyDrivenModelManager, GPUMemoryMonitor
from .tile_processor import TileProcessor
from .ai_integration import AIVideoProcessor
from .theater_audio_enhancer import TheaterAudioEnhancer
from .three_stage_enhancer import ThreeStageVideoEnhancer
from .gan_mask_generator import GANSafeMaskGenerator, MaskGenerationConfig
from .temporal_lock_processor import TemporalLockProcessor, MotionDetector, TemporalStabilizationMode
from .logging_system import (
    LoggingSystem,
    PerformanceLogger,
    LogLevel,
    get_global_logging_system,
    get_logger,
    get_performance_logger,
    log_error,
    setup_logging
)
from .resource_cleanup import (
    ResourceCleanupManager,
    TemporaryFileManager,
    get_global_cleanup_manager,
    register_for_cleanup,
    cleanup_resource
)

__all__ = [
    "VideoProcessor",
    "AudioEnhancer",
    "MemoryManager", 
    "ProgressTracker",
    "ConfigurationManager",
    "CheckpointSystem",
    "ErrorHandler",
    "AIModelManagerInterface",
    "TileProcessorInterface",
    "ConservativeMemoryManager",
    "MultiStageProgressTracker",
    "AIModelManager",
    "StrategyDrivenModelManager",
    "GPUMemoryMonitor",
    "TileProcessor",
    "AIVideoProcessor",
    "TheaterAudioEnhancer",
    "ThreeStageVideoEnhancer",
    "GANSafeMaskGenerator",
    "MaskGenerationConfig",
    "TemporalLockProcessor",
    "MotionDetector",
    "TemporalStabilizationMode",
    "LoggingSystem",
    "PerformanceLogger",
    "LogLevel",
    "get_global_logging_system",
    "get_logger",
    "get_performance_logger",
    "log_error",
    "setup_logging",
    "ResourceCleanupManager",
    "TemporaryFileManager",
    "get_global_cleanup_manager",
    "register_for_cleanup",
    "cleanup_resource"
]