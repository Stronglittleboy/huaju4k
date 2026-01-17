"""
Recovery Coordinator

High-level coordinator for error recovery and checkpoint management during processing.
"""

from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import asyncio
from pathlib import Path

from .interfaces import ILogger, IProgressTracker, IErrorHandler
from .error_models import (
    ErrorReport, ErrorContext, CheckpointData, RecoveryAction,
    VideoEnhancementError, ProcessingError
)
from .error_reporter import UserErrorReporter


class ProcessingRecoveryCoordinator:
    """Coordinates error recovery and checkpoint management during processing operations."""
    
    def __init__(self, error_handler: IErrorHandler, logger: ILogger, 
                 progress_tracker: IProgressTracker, error_reporter: UserErrorReporter):
        self.error_handler = error_handler
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.error_reporter = error_reporter
        
        # Processing state
        self.current_operation: Optional[str] = None
        self.current_stage: Optional[str] = None
        self.processing_state: Dict[str, Any] = {}
        self.intermediate_results: Dict[str, Any] = {}
        
        # Recovery settings
        self.auto_checkpoint_interval = 300  # 5 minutes
        self.max_retry_attempts = 3
        self.interactive_recovery = True
        
        # Checkpoint management
        self.last_checkpoint_time = datetime.now()
        self.checkpoint_counter = 0
    
    async def execute_with_recovery(self, operation: str, stage: str, 
                                  processing_func: Callable, *args, **kwargs) -> Any:
        """Execute a processing function with comprehensive error recovery."""
        self.current_operation = operation
        self.current_stage = stage
        
        # Create error context
        context = ErrorContext(
            operation=operation,
            component=stage,
            input_data={"args": args, "kwargs": kwargs},
            system_state=self.processing_state.copy(),
            timestamp=datetime.now()
        )
        
        attempt = 0
        last_error = None
        
        while attempt < self.max_retry_attempts:
            try:
                # Check if we should create a checkpoint
                await self._maybe_create_checkpoint()
                
                # Execute the processing function
                self.logger.log_operation(
                    "processing_start",
                    {
                        "operation": operation,
                        "stage": stage,
                        "attempt": attempt + 1
                    }
                )
                
                result = await self._execute_function(processing_func, *args, **kwargs)
                
                # Success - log and return
                self.logger.log_operation(
                    "processing_success",
                    {
                        "operation": operation,
                        "stage": stage,
                        "attempt": attempt + 1
                    }
                )
                
                return result
                
            except Exception as error:
                attempt += 1
                last_error = error
                
                # Handle the error
                error_report = self.error_handler.handle_error(error, context)
                
                # Display error to user if interactive
                if self.interactive_recovery:
                    self.error_reporter.display_error(error_report)
                
                # Determine recovery action
                recovery_action = await self._determine_recovery_action(error_report, attempt)
                
                if recovery_action is None or recovery_action == RecoveryAction.ABORT:
                    # Create emergency checkpoint before aborting
                    await self._create_emergency_checkpoint(error_report)
                    raise error
                
                # Execute recovery action
                recovery_success = self.error_handler.execute_recovery(error_report, recovery_action)
                
                if self.interactive_recovery:
                    self.error_reporter.display_recovery_result(
                        recovery_action, recovery_success
                    )
                
                if not recovery_success:
                    continue
                
                # Handle specific recovery actions
                if recovery_action == RecoveryAction.SKIP:
                    self.logger.log_operation(
                        "processing_skipped",
                        {"operation": operation, "stage": stage, "reason": "error_recovery"}
                    )
                    return None  # Indicate skipped
                
                elif recovery_action == RecoveryAction.FALLBACK:
                    # Apply fallback parameters and retry
                    kwargs = self._apply_fallback_parameters(kwargs, error_report)
                    continue
                
                elif recovery_action == RecoveryAction.RETRY:
                    # Just continue the loop for retry
                    continue
        
        # All attempts failed
        if last_error:
            await self._create_emergency_checkpoint(
                self.error_handler.handle_error(last_error, context)
            )
            raise ProcessingError(
                f"Processing failed after {self.max_retry_attempts} attempts: {last_error}",
                operation=operation,
                stage=stage,
                context=context
            )
    
    async def resume_from_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Resume processing from a saved checkpoint."""
        try:
            checkpoint = self.error_handler.load_checkpoint(checkpoint_id)
            if not checkpoint:
                raise ProcessingError(
                    f"Checkpoint {checkpoint_id} not found",
                    operation="resume_checkpoint"
                )
            
            # Restore processing state
            self.current_operation = checkpoint.operation
            self.current_stage = checkpoint.stage
            self.processing_state = checkpoint.state_data.copy()
            self.intermediate_results = checkpoint.intermediate_results.copy()
            
            self.logger.log_operation(
                "checkpoint_resumed",
                {
                    "checkpoint_id": checkpoint_id,
                    "operation": checkpoint.operation,
                    "stage": checkpoint.stage,
                    "progress": checkpoint.progress
                }
            )
            
            return {
                "operation": checkpoint.operation,
                "stage": checkpoint.stage,
                "progress": checkpoint.progress,
                "state_data": checkpoint.state_data,
                "intermediate_results": checkpoint.intermediate_results
            }
            
        except Exception as error:
            self.logger.log_error(error, {"checkpoint_id": checkpoint_id})
            raise ProcessingError(
                f"Failed to resume from checkpoint {checkpoint_id}: {error}",
                operation="resume_checkpoint"
            )
    
    def update_processing_state(self, key: str, value: Any) -> None:
        """Update the current processing state."""
        self.processing_state[key] = value
        self.logger.log_operation(
            "state_updated",
            {"key": key, "operation": self.current_operation, "stage": self.current_stage}
        )
    
    def add_intermediate_result(self, key: str, result: Any) -> None:
        """Add an intermediate processing result."""
        self.intermediate_results[key] = result
        self.logger.log_operation(
            "intermediate_result_added",
            {"key": key, "operation": self.current_operation, "stage": self.current_stage}
        )
    
    def set_checkpoint_interval(self, seconds: int) -> None:
        """Set the automatic checkpoint interval."""
        self.auto_checkpoint_interval = seconds
        self.logger.log_operation(
            "checkpoint_interval_updated",
            {"interval_seconds": seconds}
        )
    
    def enable_interactive_recovery(self, enabled: bool = True) -> None:
        """Enable or disable interactive error recovery."""
        self.interactive_recovery = enabled
        self.logger.log_operation(
            "interactive_recovery_updated",
            {"enabled": enabled}
        )
    
    async def list_available_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints for resuming."""
        checkpoints = self.error_handler.list_checkpoints()
        
        checkpoint_info = []
        for checkpoint in checkpoints:
            checkpoint_info.append({
                "checkpoint_id": checkpoint.checkpoint_id,
                "operation": checkpoint.operation,
                "stage": checkpoint.stage,
                "progress": checkpoint.progress,
                "timestamp": checkpoint.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "age_hours": (datetime.now() - checkpoint.timestamp).total_seconds() / 3600
            })
        
        return checkpoint_info
    
    async def cleanup_old_checkpoints(self, older_than_hours: int = 24) -> int:
        """Clean up old checkpoint files."""
        removed_count = self.error_handler.cleanup_checkpoints(older_than_hours)
        
        self.logger.log_operation(
            "checkpoints_cleaned",
            {"removed_count": removed_count, "older_than_hours": older_than_hours}
        )
        
        return removed_count
    
    # Private helper methods
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function, handling both sync and async functions."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    async def _maybe_create_checkpoint(self) -> None:
        """Create a checkpoint if the interval has elapsed."""
        now = datetime.now()
        elapsed = (now - self.last_checkpoint_time).total_seconds()
        
        if elapsed >= self.auto_checkpoint_interval:
            await self._create_checkpoint("automatic")
    
    async def _create_checkpoint(self, reason: str) -> CheckpointData:
        """Create a processing checkpoint."""
        if not self.current_operation or not self.current_stage:
            return None
        
        self.checkpoint_counter += 1
        
        # Calculate progress (this would be more sophisticated in real implementation)
        progress = min(0.1 * self.checkpoint_counter, 0.9)  # Simplified progress calculation
        
        checkpoint = self.error_handler.create_checkpoint(
            operation=self.current_operation,
            stage=self.current_stage,
            progress=progress,
            state_data=self.processing_state.copy(),
            intermediate_results=self.intermediate_results.copy()
        )
        
        # Save the checkpoint
        self.error_handler.save_checkpoint(checkpoint)
        
        # Update tracking
        self.last_checkpoint_time = datetime.now()
        
        # Display to user if interactive
        if self.interactive_recovery:
            self.error_reporter.display_checkpoint_info(
                checkpoint.checkpoint_id,
                checkpoint.operation,
                checkpoint.stage,
                checkpoint.progress
            )
        
        self.logger.log_operation(
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "reason": reason,
                "progress": progress
            }
        )
        
        return checkpoint
    
    async def _create_emergency_checkpoint(self, error_report: ErrorReport) -> None:
        """Create an emergency checkpoint before aborting."""
        try:
            checkpoint = await self._create_checkpoint("emergency")
            if checkpoint:
                self.logger.log_operation(
                    "emergency_checkpoint_created",
                    {
                        "checkpoint_id": checkpoint.checkpoint_id,
                        "error_id": error_report.error_id
                    }
                )
        except Exception as checkpoint_error:
            self.logger.log_error(
                checkpoint_error,
                {"context": "emergency_checkpoint", "original_error": error_report.error_id}
            )
    
    async def _determine_recovery_action(self, error_report: ErrorReport, attempt: int) -> Optional[RecoveryAction]:
        """Determine the appropriate recovery action."""
        if not error_report.recovery_strategies:
            return RecoveryAction.ABORT
        
        # If not interactive, use the first available strategy
        if not self.interactive_recovery:
            return error_report.recovery_strategies[0].action
        
        # Interactive recovery - let user choose
        return self.error_reporter.get_user_recovery_choice(error_report)
    
    def _apply_fallback_parameters(self, kwargs: Dict[str, Any], error_report: ErrorReport) -> Dict[str, Any]:
        """Apply fallback parameters based on the error report."""
        # Find fallback strategy
        fallback_strategy = next(
            (s for s in error_report.recovery_strategies if s.action == RecoveryAction.FALLBACK),
            None
        )
        
        if not fallback_strategy:
            return kwargs
        
        # Apply fallback parameters
        fallback_params = fallback_strategy.parameters
        updated_kwargs = kwargs.copy()
        
        # Common fallback parameter mappings
        if fallback_params.get("reduce_threads", False):
            updated_kwargs["max_threads"] = max(1, updated_kwargs.get("max_threads", 4) // 2)
        
        if fallback_params.get("disable_gpu", False):
            updated_kwargs["use_gpu"] = False
        
        if fallback_params.get("reduce_quality", False):
            updated_kwargs["quality"] = "medium"  # Fallback to medium quality
        
        if fallback_params.get("force_cpu", False):
            updated_kwargs["force_cpu"] = True
            updated_kwargs["use_gpu"] = False
        
        if fallback_params.get("use_defaults", False):
            # Apply safe default values
            updated_kwargs.update({
                "batch_size": 1,
                "tile_size": 256,
                "use_gpu": False
            })
        
        self.logger.log_operation(
            "fallback_parameters_applied",
            {"original_params": kwargs, "fallback_params": updated_kwargs}
        )
        
        return updated_kwargs