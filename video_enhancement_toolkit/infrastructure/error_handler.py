"""
Comprehensive Error Handler

Implementation of error handling, recovery mechanisms, and checkpoint management.
"""

import os
import json
import uuid
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import logging

from .interfaces import IErrorHandler, ILogger
from .error_models import (
    ErrorReport, ErrorContext, RecoveryStrategy, CheckpointData,
    VideoEnhancementError, ErrorCategory, ErrorSeverity, RecoveryAction,
    ConfigurationError, ResourceError, ProcessingError, SystemError, ValidationError
)
from .recovery_monitor import RecoveryMonitor


class ErrorRecoveryManager(IErrorHandler):
    """Comprehensive error handling and recovery system."""
    
    def __init__(self, logger: ILogger, checkpoint_dir: str = "checkpoints"):
        self.logger = logger
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Recovery strategy registry
        self._recovery_strategies: Dict[ErrorCategory, List[RecoveryStrategy]] = {
            ErrorCategory.CONFIGURATION: [
                RecoveryStrategy(
                    action=RecoveryAction.FALLBACK,
                    description="Use default configuration values",
                    parameters={"use_defaults": True},
                    max_attempts=1
                ),
                RecoveryStrategy(
                    action=RecoveryAction.ABORT,
                    description="Stop processing and request user intervention",
                    parameters={"save_state": True},
                    max_attempts=1
                )
            ],
            ErrorCategory.RESOURCE: [
                RecoveryStrategy(
                    action=RecoveryAction.FALLBACK,
                    description="Reduce resource usage and retry",
                    parameters={"reduce_threads": True, "disable_gpu": True},
                    max_attempts=2
                ),
                RecoveryStrategy(
                    action=RecoveryAction.RETRY,
                    description="Wait and retry with current settings",
                    parameters={"wait_seconds": 30},
                    max_attempts=3
                )
            ],
            ErrorCategory.PROCESSING: [
                RecoveryStrategy(
                    action=RecoveryAction.RETRY,
                    description="Retry processing with adjusted parameters",
                    parameters={"reduce_quality": True},
                    max_attempts=2
                ),
                RecoveryStrategy(
                    action=RecoveryAction.SKIP,
                    description="Skip problematic section and continue",
                    parameters={"mark_as_skipped": True},
                    max_attempts=1
                ),
                RecoveryStrategy(
                    action=RecoveryAction.FALLBACK,
                    description="Use CPU processing instead of GPU",
                    parameters={"force_cpu": True},
                    max_attempts=1
                )
            ],
            ErrorCategory.SYSTEM: [
                RecoveryStrategy(
                    action=RecoveryAction.RETRY,
                    description="Retry after brief delay",
                    parameters={"wait_seconds": 10},
                    max_attempts=3
                ),
                RecoveryStrategy(
                    action=RecoveryAction.ABORT,
                    description="Save state and abort processing",
                    parameters={"save_checkpoint": True},
                    max_attempts=1
                )
            ],
            ErrorCategory.VALIDATION: [
                RecoveryStrategy(
                    action=RecoveryAction.FALLBACK,
                    description="Use validated default values",
                    parameters={"use_safe_defaults": True},
                    max_attempts=1
                ),
                RecoveryStrategy(
                    action=RecoveryAction.SKIP,
                    description="Skip validation and continue with warning",
                    parameters={"log_warning": True},
                    max_attempts=1
                )
            ]
        }
        
        # Recovery action handlers
        self._recovery_handlers: Dict[RecoveryAction, Callable] = {
            RecoveryAction.RETRY: self._handle_retry,
            RecoveryAction.SKIP: self._handle_skip,
            RecoveryAction.ABORT: self._handle_abort,
            RecoveryAction.FALLBACK: self._handle_fallback,
            RecoveryAction.CONTINUE: self._handle_continue
        }
    
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> ErrorReport:
        """Handle an error and generate recovery strategies."""
        try:
            # Determine error category and severity
            if isinstance(error, VideoEnhancementError):
                category = error.category
                severity = error.severity
                error_context = error.context or context
            else:
                category, severity = self._categorize_error(error)
                error_context = context
            
            # Generate error ID
            error_id = str(uuid.uuid4())
            
            # Get recovery strategies for this error category
            recovery_strategies = self._recovery_strategies.get(category, [])
            
            # Create error report
            error_report = ErrorReport(
                error_id=error_id,
                category=category,
                severity=severity,
                message=str(error),
                technical_details=traceback.format_exc(),
                context=error_context or self._create_default_context(),
                recovery_strategies=recovery_strategies
            )
            
            # Log the error
            self.logger.log_error(error, {
                "error_id": error_id,
                "category": category.value,
                "severity": severity.value,
                "context": error_context.__dict__ if error_context else None
            })
            
            return error_report
            
        except Exception as handling_error:
            # Fallback error handling
            self.logger.log_error(handling_error, {"original_error": str(error)})
            return self._create_fallback_error_report(error)
    
    def execute_recovery(self, error_report: ErrorReport, chosen_action: RecoveryAction) -> bool:
        """Execute a recovery action for an error."""
        try:
            # Find the recovery strategy for the chosen action
            strategy = next(
                (s for s in error_report.recovery_strategies if s.action == chosen_action),
                None
            )
            
            if not strategy:
                self.logger.log_operation(
                    "recovery_execution",
                    {"error_id": error_report.error_id, "action": chosen_action.value, "status": "no_strategy"},
                    "WARNING"
                )
                return False
            
            # Execute the recovery action
            handler = self._recovery_handlers.get(chosen_action)
            if not handler:
                self.logger.log_operation(
                    "recovery_execution",
                    {"error_id": error_report.error_id, "action": chosen_action.value, "status": "no_handler"},
                    "ERROR"
                )
                return False
            
            success = handler(error_report, strategy)
            
            # Log recovery attempt
            self.logger.log_operation(
                "recovery_execution",
                {
                    "error_id": error_report.error_id,
                    "action": chosen_action.value,
                    "strategy": strategy.description,
                    "success": success
                }
            )
            
            # Update error report if successful
            if success:
                error_report.resolution = f"Resolved using {chosen_action.value}: {strategy.description}"
                error_report.resolved_at = datetime.now()
            
            return success
            
        except Exception as recovery_error:
            self.logger.log_error(recovery_error, {
                "error_id": error_report.error_id,
                "recovery_action": chosen_action.value
            })
            return False
    
    def create_checkpoint(self, operation: str, stage: str, progress: float,
                         state_data: Dict[str, Any], intermediate_results: Dict[str, Any]) -> CheckpointData:
        """Create a processing checkpoint."""
        checkpoint_id = f"{operation}_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        checkpoint = CheckpointData(
            checkpoint_id=checkpoint_id,
            operation=operation,
            stage=stage,
            progress=progress,
            state_data=state_data,
            intermediate_results=intermediate_results,
            timestamp=datetime.now(),
            metadata={
                "created_by": "ErrorRecoveryManager",
                "system_info": self._get_system_info()
            }
        )
        
        self.logger.log_operation(
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint_id,
                "operation": operation,
                "stage": stage,
                "progress": progress
            }
        )
        
        return checkpoint
    
    def save_checkpoint(self, checkpoint: CheckpointData) -> None:
        """Save checkpoint data to persistent storage."""
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.json"
            
            # Convert checkpoint to serializable format
            checkpoint_dict = {
                "checkpoint_id": checkpoint.checkpoint_id,
                "operation": checkpoint.operation,
                "stage": checkpoint.stage,
                "progress": checkpoint.progress,
                "state_data": checkpoint.state_data,
                "intermediate_results": checkpoint.intermediate_results,
                "timestamp": checkpoint.timestamp.isoformat(),
                "metadata": checkpoint.metadata
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_dict, f, indent=2, default=str)
            
            self.logger.log_operation(
                "checkpoint_saved",
                {"checkpoint_id": checkpoint.checkpoint_id, "file": str(checkpoint_file)}
            )
            
        except Exception as save_error:
            self.logger.log_error(save_error, {"checkpoint_id": checkpoint.checkpoint_id})
            raise ProcessingError(
                f"Failed to save checkpoint {checkpoint.checkpoint_id}: {save_error}",
                operation="save_checkpoint"
            )
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointData]:
        """Load checkpoint data from persistent storage."""
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
            
            if not checkpoint_file.exists():
                return None
            
            with open(checkpoint_file, 'r') as f:
                checkpoint_dict = json.load(f)
            
            checkpoint = CheckpointData(
                checkpoint_id=checkpoint_dict["checkpoint_id"],
                operation=checkpoint_dict["operation"],
                stage=checkpoint_dict["stage"],
                progress=checkpoint_dict["progress"],
                state_data=checkpoint_dict["state_data"],
                intermediate_results=checkpoint_dict["intermediate_results"],
                timestamp=datetime.fromisoformat(checkpoint_dict["timestamp"]),
                metadata=checkpoint_dict["metadata"]
            )
            
            self.logger.log_operation(
                "checkpoint_loaded",
                {"checkpoint_id": checkpoint_id}
            )
            
            return checkpoint
            
        except Exception as load_error:
            self.logger.log_error(load_error, {"checkpoint_id": checkpoint_id})
            return None
    
    def list_checkpoints(self, operation: Optional[str] = None) -> List[CheckpointData]:
        """List available checkpoints."""
        checkpoints = []
        
        try:
            for checkpoint_file in self.checkpoint_dir.glob("*.json"):
                try:
                    with open(checkpoint_file, 'r') as f:
                        checkpoint_dict = json.load(f)
                    
                    # Filter by operation if specified
                    if operation and checkpoint_dict.get("operation") != operation:
                        continue
                    
                    checkpoint = CheckpointData(
                        checkpoint_id=checkpoint_dict["checkpoint_id"],
                        operation=checkpoint_dict["operation"],
                        stage=checkpoint_dict["stage"],
                        progress=checkpoint_dict["progress"],
                        state_data=checkpoint_dict["state_data"],
                        intermediate_results=checkpoint_dict["intermediate_results"],
                        timestamp=datetime.fromisoformat(checkpoint_dict["timestamp"]),
                        metadata=checkpoint_dict["metadata"]
                    )
                    
                    checkpoints.append(checkpoint)
                    
                except Exception as parse_error:
                    self.logger.log_error(parse_error, {"checkpoint_file": str(checkpoint_file)})
                    continue
            
            # Sort by timestamp (newest first)
            checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
            
        except Exception as list_error:
            self.logger.log_error(list_error, {"operation": "list_checkpoints"})
        
        return checkpoints
    
    def cleanup_checkpoints(self, older_than_hours: int = 24) -> int:
        """Clean up old checkpoint files."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            removed_count = 0
            
            for checkpoint_file in self.checkpoint_dir.glob("*.json"):
                try:
                    with open(checkpoint_file, 'r') as f:
                        checkpoint_dict = json.load(f)
                    
                    timestamp = datetime.fromisoformat(checkpoint_dict["timestamp"])
                    
                    if timestamp < cutoff_time:
                        checkpoint_file.unlink()
                        removed_count += 1
                        
                except Exception as cleanup_error:
                    self.logger.log_error(cleanup_error, {"checkpoint_file": str(checkpoint_file)})
                    continue
            
            self.logger.log_operation(
                "checkpoint_cleanup",
                {"removed_count": removed_count, "older_than_hours": older_than_hours}
            )
            
            return removed_count
            
        except Exception as cleanup_error:
            self.logger.log_error(cleanup_error, {"operation": "cleanup_checkpoints"})
            return 0
    
    # Private helper methods
    
    def _categorize_error(self, error: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """Categorize an unknown error."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Configuration-related errors
        if any(keyword in error_message for keyword in ["config", "setting", "parameter", "invalid"]):
            return ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH
        
        # Resource-related errors
        if any(keyword in error_message for keyword in ["memory", "disk", "space", "resource", "gpu", "cuda"]):
            return ErrorCategory.RESOURCE, ErrorSeverity.MEDIUM
        
        # System-related errors
        if any(keyword in error_message for keyword in ["permission", "access", "network", "connection"]):
            return ErrorCategory.SYSTEM, ErrorSeverity.HIGH
        
        # Processing-related errors
        if any(keyword in error_message for keyword in ["process", "encode", "decode", "frame", "audio", "video"]):
            return ErrorCategory.PROCESSING, ErrorSeverity.HIGH
        
        # Default to system error
        return ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM
    
    def _create_default_context(self) -> ErrorContext:
        """Create a default error context."""
        return ErrorContext(
            operation="unknown",
            component="unknown",
            input_data={},
            system_state=self._get_system_info(),
            timestamp=datetime.now()
        )
    
    def _create_fallback_error_report(self, error: Exception) -> ErrorReport:
        """Create a fallback error report when error handling fails."""
        return ErrorReport(
            error_id=str(uuid.uuid4()),
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            message=f"Error handling failed: {str(error)}",
            technical_details=traceback.format_exc(),
            context=self._create_default_context(),
            recovery_strategies=[
                RecoveryStrategy(
                    action=RecoveryAction.ABORT,
                    description="Abort processing due to critical error",
                    parameters={"save_logs": True},
                    max_attempts=1
                )
            ]
        )
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        import platform
        import psutil
        
        try:
            return {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_free": psutil.disk_usage('.').free
            }
        except Exception:
            return {"error": "Could not retrieve system info"}
    
    # Recovery action handlers
    
    def _handle_retry(self, error_report: ErrorReport, strategy: RecoveryStrategy) -> bool:
        """Handle retry recovery action."""
        wait_seconds = strategy.parameters.get("wait_seconds", 0)
        if wait_seconds > 0:
            import time
            time.sleep(wait_seconds)
        return True  # Indicate that retry should be attempted
    
    def _handle_skip(self, error_report: ErrorReport, strategy: RecoveryStrategy) -> bool:
        """Handle skip recovery action."""
        if strategy.parameters.get("mark_as_skipped", False):
            self.logger.log_operation(
                "processing_skipped",
                {"error_id": error_report.error_id, "reason": "error_recovery"}
            )
        return True
    
    def _handle_abort(self, error_report: ErrorReport, strategy: RecoveryStrategy) -> bool:
        """Handle abort recovery action."""
        if strategy.parameters.get("save_checkpoint", False):
            # This would be handled by the calling code
            pass
        if strategy.parameters.get("save_logs", False):
            # This would be handled by the logger
            pass
        return True
    
    def _handle_fallback(self, error_report: ErrorReport, strategy: RecoveryStrategy) -> bool:
        """Handle fallback recovery action."""
        # Fallback parameters would be applied by the calling code
        return True
    
    def _handle_continue(self, error_report: ErrorReport, strategy: RecoveryStrategy) -> bool:
        """Handle continue recovery action."""
        return True