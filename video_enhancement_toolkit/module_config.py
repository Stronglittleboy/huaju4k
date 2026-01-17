"""
Module Configuration

Configures dependency injection container with default implementations.
"""

from .container import container
from .core.interfaces import IVideoProcessor, IAudioOptimizer, IPerformanceManager
from .core.models import VideoConfig, AudioConfig
from .core.audio_optimizer import TheaterAudioOptimizer
from .core.video_processor import VideoProcessor
from .infrastructure.interfaces import ILogger, IProgressTracker, IConfigurationManager
from .infrastructure.configuration_manager import ConfigurationManager
from .infrastructure.structured_logger import StructuredLogger
from .infrastructure.progress_tracker import RichProgressTracker
from .infrastructure.performance_manager import PerformanceManager
from .infrastructure.models import LoggingConfig, LogLevel
from .cli.interfaces import ICLIController, IMenuSystem
from .cli.cli_controller import VideoEnhancementCLI
from .cli.menu_system import MenuSystem


def _create_default_logging_config() -> LoggingConfig:
    """Create default logging configuration."""
    return LoggingConfig(
        log_level=LogLevel.INFO,
        log_file_path="video_enhancement.log",
        console_output=True,
        json_format=True,
        max_file_size=10 * 1024 * 1024,  # 10MB
        backup_count=5
    )


def _create_default_video_config() -> VideoConfig:
    """Create default video processing configuration."""
    return VideoConfig(
        input_path="",
        output_path="",
        target_resolution=(3840, 2160),
        ai_model="RealESRGAN_x4plus",
        tile_size=640,
        batch_size=1,
        gpu_acceleration=True
    )


def configure_default_modules():
    """Configure container with default module implementations.
    
    This function sets up the dependency injection container with default
    implementations for all interfaces. Implementations will be created
    in subsequent tasks.
    """
    # Infrastructure modules - register with factory to avoid constructor dependencies
    container.register_factory(IConfigurationManager, lambda: ConfigurationManager())
    container.register_factory(
        ILogger, 
        lambda: StructuredLogger(_create_default_logging_config())
    )
    container.register_factory(IProgressTracker, lambda: RichProgressTracker(container.resolve(ILogger)))
    container.register_factory(IPerformanceManager, lambda: PerformanceManager(container.resolve(ILogger)))
    
    # CLI modules
    container.register_factory(IMenuSystem, lambda: MenuSystem())
    container.register_factory(
        ICLIController, 
        lambda: VideoEnhancementCLI(
            container.resolve(IConfigurationManager),
            container.resolve(ILogger),
            container.resolve(IProgressTracker),
            container.resolve(IMenuSystem)
        )
    )
    
    # Core processing modules
    container.register_factory(
        IVideoProcessor, 
        lambda: VideoProcessor(
            _create_default_video_config(),
            container.resolve(ILogger),
            container.resolve(IProgressTracker)
        )
    )
    container.register_factory(
        IAudioOptimizer,
        lambda: TheaterAudioOptimizer(
            AudioConfig(
                noise_reduction_strength=0.3,
                dialogue_enhancement=0.4,
                dynamic_range_target=-23.0,
                preserve_naturalness=True,
                theater_preset="medium"
            ),
            container.resolve(ILogger)
        )
    )


def configure_test_modules():
    """Configure container with test/mock implementations.
    
    This function can be used during testing to inject mock implementations.
    """
    pass


def get_configured_container():
    """Get a configured dependency injection container.
    
    Returns:
        Configured DIContainer instance
    """
    configure_default_modules()
    return container