"""
Error Handling Models

Data models and exception classes for comprehensive error handling.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime


class ErrorCategory(Enum):
    """Categories of errors that can occur in the system."""
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    PROCESSING = "processing"
    SYSTEM = "system"
    VALIDATION = "validation"
    NETWORK = "network"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Available recovery actions for errors."""
    RETRY = "retry"
    SKIP = "skip"
    ABORT = "abort"
    FALLBACK = "fallback"
    CONTINUE = "continue"


@dataclass
class ErrorContext:
    """Context information for an error."""
    operation: str
    component: str
    input_data: Dict[str, Any]
    system_state: Dict[str, Any]
    timestamp: datetime
    user_action: Optional[str] = None


@dataclass
class RecoveryStrategy:
    """Strategy for recovering from an error."""
    action: RecoveryAction
    description: str
    parameters: Dict[str, Any]
    max_attempts: int = 3
    fallback_strategy: Optional['RecoveryStrategy'] = None


@dataclass
class ErrorReport:
    """Comprehensive error report."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    technical_details: str
    context: ErrorContext
    recovery_strategies: List[RecoveryStrategy]
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None


@dataclass
class CheckpointData:
    """Data structure for processing checkpoints."""
    checkpoint_id: str
    operation: str
    stage: str
    progress: float
    state_data: Dict[str, Any]
    intermediate_results: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]


# Custom Exception Classes

class VideoEnhancementError(Exception):
    """Base exception for video enhancement toolkit."""
    
    def __init__(self, message: str, category: ErrorCategory, severity: ErrorSeverity, 
                 context: Optional[ErrorContext] = None, recovery_strategies: Optional[List[RecoveryStrategy]] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.recovery_strategies = recovery_strategies or []
        self.timestamp = datetime.now()


class ConfigurationError(VideoEnhancementError):
    """Error related to configuration issues."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            context=context
        )
        self.config_key = config_key


class ResourceError(VideoEnhancementError):
    """Error related to system resource issues."""
    
    def __init__(self, message: str, resource_type: str, 
                 context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )
        self.resource_type = resource_type


class ProcessingError(VideoEnhancementError):
    """Error during video/audio processing operations."""
    
    def __init__(self, message: str, operation: str, stage: Optional[str] = None,
                 context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.HIGH,
            context=context
        )
        self.operation = operation
        self.stage = stage


class SystemError(VideoEnhancementError):
    """Error related to system-level issues."""
    
    def __init__(self, message: str, system_component: str,
                 context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            context=context
        )
        self.system_component = system_component


class ValidationError(VideoEnhancementError):
    """Error during data validation."""
    
    def __init__(self, message: str, validation_type: str, invalid_value: Any = None,
                 context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )
        self.validation_type = validation_type
        self.invalid_value = invalid_value