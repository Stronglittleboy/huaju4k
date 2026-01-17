"""
Infrastructure Layer

Contains logging, progress tracking, configuration management, and other infrastructure components.
"""

from .interfaces import ILogger, IProgressTracker, IConfigurationManager, IErrorHandler
from .models import LoggingConfig, ProgressTask, ConfigurationData, LogLevel
from .error_models import (
    ErrorCategory, ErrorSeverity, RecoveryAction, ErrorContext,
    RecoveryStrategy, ErrorReport, CheckpointData,
    VideoEnhancementError, ConfigurationError, ResourceError,
    ProcessingError, SystemError, ValidationError
)
from .configuration_manager import ConfigurationManager
from .config_validator import ConfigValidator
from .config_cli import ConfigCLI
from .structured_logger import StructuredLogger, create_logger
from .progress_tracker import RichProgressTracker, create_progress_tracker
from .error_handler import ErrorRecoveryManager
from .error_reporter import UserErrorReporter
from .recovery_coordinator import ProcessingRecoveryCoordinator
from .safe_processing import (
    SafeProcessingContext, safe_processing, safe_processing_sync,
    ProcessingPipeline, BatchProcessor
)
from .performance_manager import (
    PerformanceManager, 
    SystemResources, 
    PerformanceMetrics, 
    ResourceAllocation,
    AdaptiveResourceManager,
    MemoryManager,
    BottleneckDetector,
    MemoryPool
)

__all__ = [
    "ILogger",
    "IProgressTracker", 
    "IConfigurationManager",
    "IErrorHandler",
    "LoggingConfig",
    "ProgressTask",
    "ConfigurationData",
    "LogLevel",
    # Error Models
    "ErrorCategory", "ErrorSeverity", "RecoveryAction", "ErrorContext",
    "RecoveryStrategy", "ErrorReport", "CheckpointData",
    "VideoEnhancementError", "ConfigurationError", "ResourceError",
    "ProcessingError", "SystemError", "ValidationError",
    # Implementations
    "ConfigurationManager",
    "ConfigValidator",
    "ConfigCLI",
    "StructuredLogger",
    "create_logger",
    "RichProgressTracker",
    "create_progress_tracker",
    "ErrorRecoveryManager",
    "UserErrorReporter",
    "ProcessingRecoveryCoordinator",
    "SafeProcessingContext",
    "safe_processing",
    "safe_processing_sync", 
    "ProcessingPipeline",
    "BatchProcessor",
    "PerformanceManager",
    "SystemResources",
    "PerformanceMetrics",
    "ResourceAllocation",
    "AdaptiveResourceManager",
    "MemoryManager",
    "BottleneckDetector",
    "MemoryPool"
]