"""
Infrastructure Interfaces

Abstract base classes for infrastructure components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from .models import LoggingConfig, ProgressTask, ConfigurationData
from .error_models import (
    ErrorReport, ErrorContext, RecoveryStrategy, CheckpointData,
    VideoEnhancementError, RecoveryAction
)


class ILogger(ABC):
    """Interface for structured logging system."""
    
    @abstractmethod
    def log_operation(self, operation: str, details: Dict[str, Any], level: str = "INFO") -> None:
        """Log structured operation data.
        
        Args:
            operation: Name of the operation
            details: Dictionary with operation details
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        pass
    
    @abstractmethod
    def log_performance(self, metrics: Dict[str, Any]) -> None:
        """Log performance metrics.
        
        Args:
            metrics: Dictionary with performance metrics
        """
        pass
    
    @abstractmethod
    def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log detailed error information.
        
        Args:
            error: Exception that occurred
            context: Dictionary with error context
        """
        pass
    
    @abstractmethod
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive processing report.
        
        Returns:
            Dictionary with processing report data
        """
        pass


class IProgressTracker(ABC):
    """Interface for progress tracking system."""
    
    @abstractmethod
    def start_task(self, task_id: str, total_steps: int, description: str) -> None:
        """Start tracking a new task.
        
        Args:
            task_id: Unique identifier for the task
            total_steps: Total number of steps in the task
            description: Human-readable task description
        """
        pass
    
    @abstractmethod
    def update_progress(self, task_id: str, completed_steps: int, status_message: str = "") -> None:
        """Update task progress.
        
        Args:
            task_id: Task identifier
            completed_steps: Number of completed steps
            status_message: Optional status message
        """
        pass
    
    @abstractmethod
    def complete_task(self, task_id: str, success: bool, summary: str) -> None:
        """Mark task as completed.
        
        Args:
            task_id: Task identifier
            success: Whether task completed successfully
            summary: Summary of task completion
        """
        pass
    
    @abstractmethod
    def get_task_status(self, task_id: str) -> Optional[ProgressTask]:
        """Get current status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            ProgressTask object or None if task not found
        """
        pass


class IErrorHandler(ABC):
    """Interface for comprehensive error handling and recovery."""
    
    @abstractmethod
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> ErrorReport:
        """Handle an error and generate recovery strategies.
        
        Args:
            error: Exception that occurred
            context: Optional error context
            
        Returns:
            ErrorReport with analysis and recovery strategies
        """
        pass
    
    @abstractmethod
    def execute_recovery(self, error_report: ErrorReport, chosen_action: RecoveryAction) -> bool:
        """Execute a recovery action for an error.
        
        Args:
            error_report: Error report with recovery strategies
            chosen_action: Recovery action to execute
            
        Returns:
            True if recovery was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def create_checkpoint(self, operation: str, stage: str, progress: float, 
                         state_data: Dict[str, Any], intermediate_results: Dict[str, Any]) -> CheckpointData:
        """Create a processing checkpoint.
        
        Args:
            operation: Name of the operation
            stage: Current processing stage
            progress: Progress percentage (0.0-1.0)
            state_data: Current processing state
            intermediate_results: Intermediate processing results
            
        Returns:
            CheckpointData object
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, checkpoint: CheckpointData) -> None:
        """Save checkpoint data to persistent storage.
        
        Args:
            checkpoint: Checkpoint data to save
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointData]:
        """Load checkpoint data from persistent storage.
        
        Args:
            checkpoint_id: Identifier of the checkpoint to load
            
        Returns:
            CheckpointData object or None if not found
        """
        pass
    
    @abstractmethod
    def list_checkpoints(self, operation: Optional[str] = None) -> List[CheckpointData]:
        """List available checkpoints.
        
        Args:
            operation: Optional operation filter
            
        Returns:
            List of available checkpoints
        """
        pass
    
    @abstractmethod
    def cleanup_checkpoints(self, older_than_hours: int = 24) -> int:
        """Clean up old checkpoint files.
        
        Args:
            older_than_hours: Remove checkpoints older than this many hours
            
        Returns:
            Number of checkpoints removed
        """
        pass


class IConfigurationManager(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    def load_configuration(self, config_path: Optional[str] = None) -> ConfigurationData:
        """Load configuration from file.
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            ConfigurationData object with loaded configuration
        """
        pass
    
    @abstractmethod
    def save_configuration(self, config: ConfigurationData, config_path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration data to save
            config_path: Optional path to save configuration
        """
        pass
    
    @abstractmethod
    def get_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration preset by name.
        
        Args:
            preset_name: Name of the preset
            
        Returns:
            Dictionary with preset configuration or None if not found
        """
        pass
    
    @abstractmethod
    def save_preset(self, preset_name: str, config: Dict[str, Any]) -> None:
        """Save configuration as preset.
        
        Args:
            preset_name: Name for the preset
            config: Configuration data to save as preset
        """
        pass
    
    @abstractmethod
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        pass