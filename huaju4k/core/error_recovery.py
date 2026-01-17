"""
Error handling and recovery system for huaju4k video enhancement.

This module provides comprehensive error classification, automatic recovery mechanisms,
and diagnostic information generation for robust video processing operations.
"""

import os
import logging
import traceback
import time
from typing import Dict, List, Optional, Any, Callable, Type
from pathlib import Path
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from ..models.data_models import CheckpointData, ProcessingStrategy
from .checkpoint_system import CheckpointSystem

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors that can occur during processing."""
    
    SYSTEM = "system"           # System-level errors (memory, disk, etc.)
    INPUT = "input"             # Input file or parameter errors
    PROCESSING = "processing"   # Processing algorithm errors
    OUTPUT = "output"           # Output generation errors
    NETWORK = "network"         # Network-related errors
    DEPENDENCY = "dependency"   # Missing dependencies or libraries
    CONFIGURATION = "config"    # Configuration or setup errors
    UNKNOWN = "unknown"         # Unclassified errors


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    
    LOW = "low"                 # Minor issues, processing can continue
    MEDIUM = "medium"           # Significant issues, may need intervention
    HIGH = "high"               # Critical issues, processing should stop
    FATAL = "fatal"             # Unrecoverable errors


class RecoveryAction(Enum):
    """Types of recovery actions that can be taken."""
    
    RETRY = "retry"                     # Retry the failed operation
    SKIP = "skip"                       # Skip the failed operation
    FALLBACK = "fallback"               # Use fallback method
    RESTART_STAGE = "restart_stage"     # Restart current processing stage
    RESTORE_CHECKPOINT = "restore"      # Restore from checkpoint
    USER_INTERVENTION = "user"          # Require user intervention
    ABORT = "abort"                     # Abort processing


@dataclass
class ErrorContext:
    """Context information for an error."""
    
    operation: str                      # Operation being performed
    stage: str                          # Current processing stage
    input_path: str                     # Input file path
    output_path: Optional[str] = None   # Output file path
    progress: float = 0.0               # Progress when error occurred
    system_info: Dict[str, Any] = field(default_factory=dict)
    processing_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryStrategy:
    """Strategy for recovering from an error."""
    
    primary_action: RecoveryAction      # Primary recovery action
    fallback_actions: List[RecoveryAction] = field(default_factory=list)
    max_retries: int = 3                # Maximum retry attempts
    retry_delay: float = 1.0            # Delay between retries (seconds)
    requires_user_input: bool = False   # Whether user input is required
    checkpoint_before_retry: bool = True # Create checkpoint before retry


@dataclass
class ErrorReport:
    """Comprehensive error report."""
    
    error_id: str                       # Unique error identifier
    timestamp: datetime                 # When error occurred
    category: ErrorCategory             # Error category
    severity: ErrorSeverity             # Error severity
    message: str                        # Error message
    exception_type: str                 # Exception type name
    traceback: str                      # Full traceback
    context: ErrorContext               # Error context
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_attempts: List[Dict[str, Any]] = field(default_factory=list)
    resolved: bool = False              # Whether error was resolved
    resolution_method: Optional[str] = None
    
    def add_recovery_attempt(self, action: RecoveryAction, success: bool, 
                           details: Optional[str] = None) -> None:
        """Add a recovery attempt to the report."""
        attempt = {
            'timestamp': datetime.now(),
            'action': action.value,
            'success': success,
            'details': details or ""
        }
        self.recovery_attempts.append(attempt)


class ErrorClassifier:
    """Classifies errors and determines appropriate recovery strategies."""
    
    def __init__(self):
        """Initialize error classifier with classification rules."""
        self.classification_rules = self._build_classification_rules()
        self.recovery_strategies = self._build_recovery_strategies()
    
    def classify_error(self, exception: Exception, context: ErrorContext) -> tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify an error based on exception type and context.
        
        Args:
            exception: The exception that occurred
            context: Context information
            
        Returns:
            Tuple of (category, severity)
        """
        exception_type = type(exception).__name__
        error_message = str(exception).lower()
        
        # Check classification rules
        for rule in self.classification_rules:
            if self._matches_rule(exception_type, error_message, context, rule):
                return rule['category'], rule['severity']
        
        # Default classification
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM
    
    def get_recovery_strategy(self, category: ErrorCategory, severity: ErrorSeverity, 
                            context: ErrorContext) -> RecoveryStrategy:
        """
        Get recovery strategy for an error.
        
        Args:
            category: Error category
            severity: Error severity
            context: Error context
            
        Returns:
            RecoveryStrategy object
        """
        strategy_key = (category, severity)
        
        if strategy_key in self.recovery_strategies:
            base_strategy = self.recovery_strategies[strategy_key]
            
            # Customize strategy based on context
            return self._customize_strategy(base_strategy, context)
        
        # Default strategy
        return RecoveryStrategy(
            primary_action=RecoveryAction.USER_INTERVENTION,
            max_retries=1,
            requires_user_input=True
        )
    
    def _build_classification_rules(self) -> List[Dict[str, Any]]:
        """Build error classification rules."""
        return [
            # Memory errors
            {
                'exception_types': ['MemoryError', 'OutOfMemoryError'],
                'keywords': ['memory', 'out of memory', 'allocation failed'],
                'category': ErrorCategory.SYSTEM,
                'severity': ErrorSeverity.HIGH
            },
            # File I/O errors
            {
                'exception_types': ['FileNotFoundError', 'PermissionError', 'IOError'],
                'keywords': ['file not found', 'permission denied', 'no such file'],
                'category': ErrorCategory.INPUT,
                'severity': ErrorSeverity.MEDIUM
            },
            # GPU/CUDA errors
            {
                'exception_types': ['RuntimeError', 'CudaError'],
                'keywords': ['cuda', 'gpu', 'device', 'out of memory'],
                'category': ErrorCategory.SYSTEM,
                'severity': ErrorSeverity.HIGH
            },
            # Network errors
            {
                'exception_types': ['ConnectionError', 'TimeoutError', 'URLError'],
                'keywords': ['connection', 'timeout', 'network', 'download'],
                'category': ErrorCategory.NETWORK,
                'severity': ErrorSeverity.MEDIUM
            },
            # Dependency errors
            {
                'exception_types': ['ImportError', 'ModuleNotFoundError'],
                'keywords': ['import', 'module', 'package', 'dependency'],
                'category': ErrorCategory.DEPENDENCY,
                'severity': ErrorSeverity.HIGH
            },
            # Processing errors
            {
                'exception_types': ['ValueError', 'TypeError', 'IndexError'],
                'keywords': ['invalid', 'unsupported', 'format', 'codec'],
                'category': ErrorCategory.PROCESSING,
                'severity': ErrorSeverity.MEDIUM
            },
            # Configuration errors
            {
                'exception_types': ['KeyError', 'AttributeError', 'ConfigurationError'],
                'keywords': ['config', 'setting', 'parameter', 'missing'],
                'category': ErrorCategory.CONFIGURATION,
                'severity': ErrorSeverity.MEDIUM
            }
        ]
    
    def _build_recovery_strategies(self) -> Dict[tuple, RecoveryStrategy]:
        """Build recovery strategies for different error types."""
        return {
            # System errors
            (ErrorCategory.SYSTEM, ErrorSeverity.HIGH): RecoveryStrategy(
                primary_action=RecoveryAction.RESTORE_CHECKPOINT,
                fallback_actions=[RecoveryAction.FALLBACK, RecoveryAction.ABORT],
                max_retries=1,
                checkpoint_before_retry=False
            ),
            (ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM): RecoveryStrategy(
                primary_action=RecoveryAction.RETRY,
                fallback_actions=[RecoveryAction.FALLBACK],
                max_retries=2,
                retry_delay=2.0
            ),
            
            # Input errors
            (ErrorCategory.INPUT, ErrorSeverity.MEDIUM): RecoveryStrategy(
                primary_action=RecoveryAction.USER_INTERVENTION,
                fallback_actions=[RecoveryAction.SKIP],
                max_retries=1,
                requires_user_input=True
            ),
            
            # Processing errors
            (ErrorCategory.PROCESSING, ErrorSeverity.MEDIUM): RecoveryStrategy(
                primary_action=RecoveryAction.FALLBACK,
                fallback_actions=[RecoveryAction.RETRY, RecoveryAction.SKIP],
                max_retries=3,
                retry_delay=1.0
            ),
            
            # Network errors
            (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM): RecoveryStrategy(
                primary_action=RecoveryAction.RETRY,
                fallback_actions=[RecoveryAction.SKIP],
                max_retries=5,
                retry_delay=5.0
            ),
            
            # Dependency errors
            (ErrorCategory.DEPENDENCY, ErrorSeverity.HIGH): RecoveryStrategy(
                primary_action=RecoveryAction.USER_INTERVENTION,
                fallback_actions=[RecoveryAction.ABORT],
                max_retries=1,
                requires_user_input=True
            ),
            
            # Configuration errors
            (ErrorCategory.CONFIGURATION, ErrorSeverity.MEDIUM): RecoveryStrategy(
                primary_action=RecoveryAction.FALLBACK,
                fallback_actions=[RecoveryAction.USER_INTERVENTION],
                max_retries=2
            )
        }
    
    def _matches_rule(self, exception_type: str, error_message: str, 
                     context: ErrorContext, rule: Dict[str, Any]) -> bool:
        """Check if an error matches a classification rule."""
        # Check exception type
        if 'exception_types' in rule:
            if exception_type not in rule['exception_types']:
                return False
        
        # Check keywords in error message
        if 'keywords' in rule:
            if not any(keyword in error_message for keyword in rule['keywords']):
                return False
        
        return True
    
    def _customize_strategy(self, base_strategy: RecoveryStrategy, 
                          context: ErrorContext) -> RecoveryStrategy:
        """Customize recovery strategy based on context."""
        # Create a copy of the base strategy
        strategy = RecoveryStrategy(
            primary_action=base_strategy.primary_action,
            fallback_actions=base_strategy.fallback_actions.copy(),
            max_retries=base_strategy.max_retries,
            retry_delay=base_strategy.retry_delay,
            requires_user_input=base_strategy.requires_user_input,
            checkpoint_before_retry=base_strategy.checkpoint_before_retry
        )
        
        # Customize based on progress
        if context.progress > 0.8:  # Near completion
            # Prefer checkpoint restoration for late-stage errors
            if RecoveryAction.RESTORE_CHECKPOINT not in [strategy.primary_action] + strategy.fallback_actions:
                strategy.fallback_actions.insert(0, RecoveryAction.RESTORE_CHECKPOINT)
        
        # Customize based on operation type
        if 'ai_model' in context.operation.lower():
            # For AI model operations, prefer fallback to traditional methods
            if strategy.primary_action == RecoveryAction.RETRY:
                strategy.fallback_actions.insert(0, RecoveryAction.FALLBACK)
        
        return strategy


class ErrorRecoveryManager:
    """
    Manages error recovery operations for video processing.
    
    Provides automatic error recovery, checkpoint restoration,
    and diagnostic information generation.
    """
    
    def __init__(self, checkpoint_system: CheckpointSystem):
        """
        Initialize error recovery manager.
        
        Args:
            checkpoint_system: Checkpoint system for recovery operations
        """
        self.checkpoint_system = checkpoint_system
        self.classifier = ErrorClassifier()
        self.error_reports: Dict[str, ErrorReport] = {}
        self.recovery_callbacks: Dict[RecoveryAction, Callable] = {}
        
        # Statistics
        self.total_errors = 0
        self.recovered_errors = 0
        self.failed_recoveries = 0
        
        logger.info("Error recovery manager initialized")
    
    def register_recovery_callback(self, action: RecoveryAction, callback: Callable) -> None:
        """
        Register a callback for a specific recovery action.
        
        Args:
            action: Recovery action type
            callback: Callback function to execute the action
        """
        self.recovery_callbacks[action] = callback
        logger.debug(f"Registered recovery callback for {action.value}")
    
    def handle_error(self, exception: Exception, context: ErrorContext) -> Optional[ErrorReport]:
        """
        Handle an error and attempt recovery.
        
        Args:
            exception: The exception that occurred
            context: Error context information
            
        Returns:
            ErrorReport object or None if recovery failed
        """
        try:
            self.total_errors += 1
            
            # Classify the error
            category, severity = self.classifier.classify_error(exception, context)
            
            # Create error report
            error_report = self._create_error_report(exception, context, category, severity)
            
            # Store error report
            self.error_reports[error_report.error_id] = error_report
            
            logger.error(f"Error occurred: {error_report.error_id} - {category.value}/{severity.value}: {error_report.message}")
            
            # Attempt recovery
            recovery_success = self._attempt_recovery(error_report)
            
            if recovery_success:
                self.recovered_errors += 1
                error_report.resolved = True
                logger.info(f"Error recovered successfully: {error_report.error_id}")
            else:
                self.failed_recoveries += 1
                logger.error(f"Error recovery failed: {error_report.error_id}")
            
            return error_report
            
        except Exception as e:
            logger.critical(f"Error in error handling system: {e}")
            return None
    
    def get_diagnostic_info(self, error_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive diagnostic information for an error.
        
        Args:
            error_id: Error identifier
            
        Returns:
            Dictionary with diagnostic information
        """
        if error_id not in self.error_reports:
            return None
        
        error_report = self.error_reports[error_id]
        
        diagnostic_info = {
            'error_summary': {
                'id': error_report.error_id,
                'timestamp': error_report.timestamp.isoformat(),
                'category': error_report.category.value,
                'severity': error_report.severity.value,
                'message': error_report.message,
                'resolved': error_report.resolved
            },
            'context': {
                'operation': error_report.context.operation,
                'stage': error_report.context.stage,
                'progress': error_report.context.progress,
                'input_path': error_report.context.input_path,
                'output_path': error_report.context.output_path,
                'system_info': error_report.context.system_info,
                'processing_params': error_report.context.processing_params
            },
            'technical_details': {
                'exception_type': error_report.exception_type,
                'traceback': error_report.traceback
            },
            'recovery_info': {
                'strategy': {
                    'primary_action': error_report.recovery_strategy.primary_action.value if error_report.recovery_strategy else None,
                    'fallback_actions': [a.value for a in error_report.recovery_strategy.fallback_actions] if error_report.recovery_strategy else [],
                    'max_retries': error_report.recovery_strategy.max_retries if error_report.recovery_strategy else 0
                },
                'attempts': error_report.recovery_attempts,
                'resolution_method': error_report.resolution_method
            },
            'recommendations': self._generate_recommendations(error_report)
        }
        
        return diagnostic_info
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """
        Get error recovery statistics.
        
        Returns:
            Dictionary with recovery statistics
        """
        recovery_rate = (self.recovered_errors / self.total_errors * 100) if self.total_errors > 0 else 0
        
        # Count errors by category
        category_counts = {}
        severity_counts = {}
        
        for error_report in self.error_reports.values():
            category = error_report.category.value
            severity = error_report.severity.value
            
            category_counts[category] = category_counts.get(category, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_errors': self.total_errors,
            'recovered_errors': self.recovered_errors,
            'failed_recoveries': self.failed_recoveries,
            'recovery_rate_percent': round(recovery_rate, 2),
            'errors_by_category': category_counts,
            'errors_by_severity': severity_counts,
            'active_errors': len([r for r in self.error_reports.values() if not r.resolved])
        }
    
    def clear_resolved_errors(self) -> int:
        """
        Clear resolved error reports to free memory.
        
        Returns:
            Number of error reports cleared
        """
        resolved_ids = [error_id for error_id, report in self.error_reports.items() if report.resolved]
        
        for error_id in resolved_ids:
            del self.error_reports[error_id]
        
        logger.info(f"Cleared {len(resolved_ids)} resolved error reports")
        return len(resolved_ids)
    
    def _create_error_report(self, exception: Exception, context: ErrorContext,
                           category: ErrorCategory, severity: ErrorSeverity) -> ErrorReport:
        """Create a comprehensive error report."""
        error_id = self._generate_error_id(exception, context)
        
        # Get recovery strategy
        recovery_strategy = self.classifier.get_recovery_strategy(category, severity, context)
        
        error_report = ErrorReport(
            error_id=error_id,
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            message=str(exception),
            exception_type=type(exception).__name__,
            traceback=traceback.format_exc(),
            context=context,
            recovery_strategy=recovery_strategy
        )
        
        return error_report
    
    def _attempt_recovery(self, error_report: ErrorReport) -> bool:
        """Attempt to recover from an error using the recovery strategy."""
        if not error_report.recovery_strategy:
            return False
        
        strategy = error_report.recovery_strategy
        
        # Try primary action
        if self._execute_recovery_action(error_report, strategy.primary_action):
            error_report.resolution_method = strategy.primary_action.value
            return True
        
        # Try fallback actions
        for action in strategy.fallback_actions:
            if self._execute_recovery_action(error_report, action):
                error_report.resolution_method = action.value
                return True
        
        return False
    
    def _execute_recovery_action(self, error_report: ErrorReport, action: RecoveryAction) -> bool:
        """Execute a specific recovery action."""
        try:
            logger.info(f"Attempting recovery action: {action.value} for error {error_report.error_id}")
            
            success = False
            details = ""
            
            if action in self.recovery_callbacks:
                # Execute registered callback
                callback = self.recovery_callbacks[action]
                success = callback(error_report)
                details = f"Executed callback for {action.value}"
            else:
                # Built-in recovery actions
                if action == RecoveryAction.RETRY:
                    success = self._retry_operation(error_report)
                    details = "Retried failed operation"
                elif action == RecoveryAction.RESTORE_CHECKPOINT:
                    success = self._restore_from_checkpoint(error_report)
                    details = "Restored from checkpoint"
                elif action == RecoveryAction.SKIP:
                    success = True  # Always succeeds
                    details = "Skipped failed operation"
                elif action == RecoveryAction.ABORT:
                    success = False
                    details = "Aborted processing"
                else:
                    success = False
                    details = f"No handler for action: {action.value}"
            
            # Record recovery attempt
            error_report.add_recovery_attempt(action, success, details)
            
            if success:
                logger.info(f"Recovery action succeeded: {action.value}")
            else:
                logger.warning(f"Recovery action failed: {action.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing recovery action {action.value}: {e}")
            error_report.add_recovery_attempt(action, False, f"Exception: {str(e)}")
            return False
    
    def _retry_operation(self, error_report: ErrorReport) -> bool:
        """Retry the failed operation."""
        # This is a placeholder - actual retry logic would depend on the specific operation
        # In practice, this would be handled by the calling code
        logger.info(f"Retry operation requested for error {error_report.error_id}")
        return False  # Indicate that retry should be handled by caller
    
    def _restore_from_checkpoint(self, error_report: ErrorReport) -> bool:
        """Restore processing from the most recent checkpoint."""
        try:
            # Find the most recent checkpoint for this input
            checkpoints = self.checkpoint_system.list_checkpoints(error_report.context.input_path)
            
            if not checkpoints:
                logger.warning(f"No checkpoints found for {error_report.context.input_path}")
                return False
            
            # Get the most recent checkpoint
            latest_checkpoint = checkpoints[0]  # Already sorted by timestamp
            
            # Verify checkpoint integrity
            if not self.checkpoint_system.verify_checkpoint_integrity(latest_checkpoint.checkpoint_id):
                logger.error(f"Checkpoint integrity verification failed: {latest_checkpoint.checkpoint_id}")
                return False
            
            logger.info(f"Restored from checkpoint: {latest_checkpoint.checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from checkpoint: {e}")
            return False
    
    def _generate_error_id(self, exception: Exception, context: ErrorContext) -> str:
        """Generate unique error ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        operation_hash = abs(hash(f"{context.operation}_{context.stage}")) % 10000
        return f"ERR_{timestamp}_{operation_hash}"
    
    def _generate_recommendations(self, error_report: ErrorReport) -> List[str]:
        """Generate recommendations for resolving the error."""
        recommendations = []
        
        category = error_report.category
        severity = error_report.severity
        message = error_report.message.lower()
        
        # Category-specific recommendations
        if category == ErrorCategory.SYSTEM:
            if 'memory' in message:
                recommendations.extend([
                    "Reduce tile size in processing configuration",
                    "Close other applications to free memory",
                    "Consider processing video in smaller segments"
                ])
            elif 'gpu' in message or 'cuda' in message:
                recommendations.extend([
                    "Check GPU driver installation",
                    "Reduce GPU memory usage by lowering tile size",
                    "Enable CPU fallback mode"
                ])
        
        elif category == ErrorCategory.INPUT:
            recommendations.extend([
                "Verify input file exists and is accessible",
                "Check file format compatibility",
                "Ensure sufficient disk space"
            ])
        
        elif category == ErrorCategory.PROCESSING:
            recommendations.extend([
                "Try using a different AI model",
                "Adjust processing parameters",
                "Enable fallback processing mode"
            ])
        
        elif category == ErrorCategory.DEPENDENCY:
            recommendations.extend([
                "Install missing dependencies",
                "Update existing packages",
                "Check Python environment setup"
            ])
        
        # Severity-specific recommendations
        if severity == ErrorSeverity.HIGH or severity == ErrorSeverity.FATAL:
            recommendations.extend([
                "Contact support with error details",
                "Check system requirements",
                "Consider using safe mode processing"
            ])
        
        return recommendations