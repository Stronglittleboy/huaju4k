"""
Safe Processing Context Manager

Provides context managers for safe processing operations with automatic error handling and recovery.
"""

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Dict, Any, Optional, AsyncGenerator, Generator, Callable
from datetime import datetime

from .interfaces import ILogger, IProgressTracker, IErrorHandler
from .error_models import ErrorContext, ProcessingError
from .recovery_coordinator import ProcessingRecoveryCoordinator
from .error_reporter import UserErrorReporter


class SafeProcessingContext:
    """Context manager for safe processing operations."""
    
    def __init__(self, coordinator: ProcessingRecoveryCoordinator, 
                 operation: str, stage: str, **context_data):
        self.coordinator = coordinator
        self.operation = operation
        self.stage = stage
        self.context_data = context_data
        self.start_time = None
        self.success = False
    
    async def __aenter__(self):
        """Enter the async context manager."""
        self.start_time = datetime.now()
        
        # Update coordinator state
        self.coordinator.current_operation = self.operation
        self.coordinator.current_stage = self.stage
        
        # Log operation start
        self.coordinator.logger.log_operation(
            "safe_processing_start",
            {
                "operation": self.operation,
                "stage": self.stage,
                "context": self.context_data
            }
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            # Success
            self.success = True
            self.coordinator.logger.log_operation(
                "safe_processing_success",
                {
                    "operation": self.operation,
                    "stage": self.stage,
                    "duration_seconds": duration
                }
            )
        else:
            # Error occurred
            self.coordinator.logger.log_operation(
                "safe_processing_error",
                {
                    "operation": self.operation,
                    "stage": self.stage,
                    "duration_seconds": duration,
                    "error_type": exc_type.__name__,
                    "error_message": str(exc_val)
                }
            )
        
        # Don't suppress exceptions - let them propagate
        return False
    
    def update_state(self, key: str, value: Any) -> None:
        """Update processing state."""
        self.coordinator.update_processing_state(key, value)
    
    def add_result(self, key: str, result: Any) -> None:
        """Add intermediate result."""
        self.coordinator.add_intermediate_result(key, result)


@asynccontextmanager
async def safe_processing(coordinator: ProcessingRecoveryCoordinator, 
                         operation: str, stage: str, **context_data) -> AsyncGenerator[SafeProcessingContext, None]:
    """Async context manager for safe processing operations."""
    context = SafeProcessingContext(coordinator, operation, stage, **context_data)
    async with context:
        yield context


@contextmanager
def safe_processing_sync(coordinator: ProcessingRecoveryCoordinator, 
                        operation: str, stage: str, **context_data) -> Generator[SafeProcessingContext, None, None]:
    """Synchronous context manager for safe processing operations."""
    context = SafeProcessingContext(coordinator, operation, stage, **context_data)
    
    start_time = datetime.now()
    
    # Update coordinator state
    coordinator.current_operation = operation
    coordinator.current_stage = stage
    
    # Log operation start
    coordinator.logger.log_operation(
        "safe_processing_start",
        {
            "operation": operation,
            "stage": stage,
            "context": context_data
        }
    )
    
    try:
        yield context
        
        # Success
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        coordinator.logger.log_operation(
            "safe_processing_success",
            {
                "operation": operation,
                "stage": stage,
                "duration_seconds": duration
            }
        )
        
    except Exception as error:
        # Error occurred
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        coordinator.logger.log_operation(
            "safe_processing_error",
            {
                "operation": operation,
                "stage": stage,
                "duration_seconds": duration,
                "error_type": type(error).__name__,
                "error_message": str(error)
            }
        )
        
        raise  # Re-raise the exception


class ProcessingPipeline:
    """Pipeline for executing multiple processing stages with recovery."""
    
    def __init__(self, coordinator: ProcessingRecoveryCoordinator):
        self.coordinator = coordinator
        self.stages: list = []
        self.results: Dict[str, Any] = {}
    
    def add_stage(self, name: str, func: Callable, *args, **kwargs) -> 'ProcessingPipeline':
        """Add a processing stage to the pipeline."""
        self.stages.append({
            "name": name,
            "func": func,
            "args": args,
            "kwargs": kwargs
        })
        return self
    
    async def execute(self, operation_name: str) -> Dict[str, Any]:
        """Execute all stages in the pipeline with recovery."""
        total_stages = len(self.stages)
        
        for i, stage in enumerate(self.stages):
            stage_name = stage["name"]
            stage_func = stage["func"]
            stage_args = stage["args"]
            stage_kwargs = stage["kwargs"]
            
            # Update progress
            progress = i / total_stages
            self.coordinator.update_processing_state("pipeline_progress", progress)
            
            try:
                # Execute stage with recovery
                result = await self.coordinator.execute_with_recovery(
                    operation=operation_name,
                    stage=stage_name,
                    processing_func=stage_func,
                    *stage_args,
                    **stage_kwargs
                )
                
                # Store result
                self.results[stage_name] = result
                self.coordinator.add_intermediate_result(stage_name, result)
                
                # Log stage completion
                self.coordinator.logger.log_operation(
                    "pipeline_stage_completed",
                    {
                        "operation": operation_name,
                        "stage": stage_name,
                        "stage_index": i,
                        "total_stages": total_stages,
                        "progress": (i + 1) / total_stages
                    }
                )
                
            except Exception as error:
                # Log pipeline failure
                self.coordinator.logger.log_error(
                    error,
                    {
                        "operation": operation_name,
                        "failed_stage": stage_name,
                        "stage_index": i,
                        "total_stages": total_stages,
                        "completed_stages": list(self.results.keys())
                    }
                )
                raise ProcessingError(
                    f"Pipeline failed at stage '{stage_name}': {error}",
                    operation=operation_name,
                    stage=stage_name
                )
        
        # Pipeline completed successfully
        final_progress = 1.0
        self.coordinator.update_processing_state("pipeline_progress", final_progress)
        
        self.coordinator.logger.log_operation(
            "pipeline_completed",
            {
                "operation": operation_name,
                "total_stages": total_stages,
                "results_keys": list(self.results.keys())
            }
        )
        
        return self.results.copy()
    
    def get_results(self) -> Dict[str, Any]:
        """Get current pipeline results."""
        return self.results.copy()
    
    def clear_results(self) -> None:
        """Clear pipeline results."""
        self.results.clear()


class BatchProcessor:
    """Batch processor with recovery for processing multiple items."""
    
    def __init__(self, coordinator: ProcessingRecoveryCoordinator):
        self.coordinator = coordinator
        self.failed_items: list = []
        self.successful_items: list = []
        self.skipped_items: list = []
    
    async def process_batch(self, operation_name: str, items: list, 
                          processing_func: Callable, **kwargs) -> Dict[str, Any]:
        """Process a batch of items with recovery."""
        total_items = len(items)
        
        for i, item in enumerate(items):
            item_id = str(item) if not isinstance(item, dict) else item.get("id", str(i))
            
            # Update progress
            progress = i / total_items
            self.coordinator.update_processing_state("batch_progress", progress)
            
            try:
                # Process item with recovery
                result = await self.coordinator.execute_with_recovery(
                    operation=operation_name,
                    stage=f"item_{item_id}",
                    processing_func=processing_func,
                    *[item],
                    **kwargs
                )
                
                if result is None:
                    # Item was skipped
                    self.skipped_items.append(item_id)
                else:
                    # Item processed successfully
                    self.successful_items.append(item_id)
                
            except Exception as error:
                # Item failed
                self.failed_items.append(item_id)
                
                self.coordinator.logger.log_error(
                    error,
                    {
                        "operation": operation_name,
                        "item_id": item_id,
                        "item_index": i,
                        "total_items": total_items
                    }
                )
                
                # Continue with next item unless it's a critical error
                continue
        
        # Batch processing completed
        final_progress = 1.0
        self.coordinator.update_processing_state("batch_progress", final_progress)
        
        results = {
            "total_items": total_items,
            "successful": len(self.successful_items),
            "failed": len(self.failed_items),
            "skipped": len(self.skipped_items),
            "successful_items": self.successful_items.copy(),
            "failed_items": self.failed_items.copy(),
            "skipped_items": self.skipped_items.copy()
        }
        
        self.coordinator.logger.log_operation(
            "batch_processing_completed",
            {
                "operation": operation_name,
                **results
            }
        )
        
        return results
    
    def get_failed_items(self) -> list:
        """Get list of failed items."""
        return self.failed_items.copy()
    
    def get_successful_items(self) -> list:
        """Get list of successful items."""
        return self.successful_items.copy()
    
    def get_skipped_items(self) -> list:
        """Get list of skipped items."""
        return self.skipped_items.copy()
    
    def reset(self) -> None:
        """Reset batch processor state."""
        self.failed_items.clear()
        self.successful_items.clear()
        self.skipped_items.clear()