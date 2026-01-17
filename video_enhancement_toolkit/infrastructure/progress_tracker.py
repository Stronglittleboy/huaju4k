"""
Progress Tracking System

Implementation of progress tracking with Rich library for multiple concurrent
progress bars, ETA calculation, speed metrics, and real-time updates.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from threading import Lock

from rich.console import Console
from rich.progress import (
    Progress, 
    TaskID, 
    BarColumn, 
    TextColumn, 
    TimeRemainingColumn, 
    SpinnerColumn,
    MofNCompleteColumn,
    TimeElapsedColumn
)
from rich.table import Table
from rich.live import Live

from .interfaces import IProgressTracker, ILogger
from .models import ProgressTask, TaskStatus


class RichProgressTracker(IProgressTracker):
    """Progress tracking system using Rich library."""
    
    def __init__(self, logger: Optional[ILogger] = None, console: Optional[Console] = None):
        """Initialize progress tracker.
        
        Args:
            logger: Optional logger for integration
            console: Optional Rich console instance
        """
        self.logger = logger
        self.console = console or Console()
        self.tasks: Dict[str, ProgressTask] = {}
        self.rich_tasks: Dict[str, TaskID] = {}
        self.lock = Lock()
        
        # Create Rich progress instance with custom columns
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[status_message]}"),
            console=self.console,
            expand=True
        )
        
        self.live = None
        self.is_started = False
    
    def start_display(self) -> None:
        """Start the live display for progress tracking."""
        if not self.is_started:
            self.live = Live(self.progress, console=self.console, refresh_per_second=10)
            self.live.start()
            self.is_started = True
            
            if self.logger:
                self.logger.log_operation(
                    "progress_display_started",
                    {"action": "progress_display_started"}
                )
    
    def stop_display(self) -> None:
        """Stop the live display."""
        if self.is_started and self.live:
            self.live.stop()
            self.is_started = False
            
            if self.logger:
                self.logger.log_operation(
                    "progress_display_stopped",
                    {"action": "progress_display_stopped"}
                )
    
    def start_task(self, task_id: str, total_steps: int, description: str) -> None:
        """Start tracking a new task.
        
        Args:
            task_id: Unique identifier for the task
            total_steps: Total number of steps in the task
            description: Human-readable task description
        """
        with self.lock:
            # Create progress task
            task = ProgressTask(
                task_id=task_id,
                description=description,
                total_steps=total_steps,
                status=TaskStatus.IN_PROGRESS,
                start_time=datetime.now()
            )
            self.tasks[task_id] = task
            
            # Start display if not already started
            if not self.is_started:
                self.start_display()
            
            # Add to Rich progress
            rich_task_id = self.progress.add_task(
                description=description,
                total=total_steps,
                status_message=""
            )
            self.rich_tasks[task_id] = rich_task_id
            
            if self.logger:
                self.logger.log_operation(
                    "task_started",
                    {
                        "task_id": task_id,
                        "description": description,
                        "total_steps": total_steps,
                        "start_time": task.start_time.isoformat()
                    }
                )
    
    def update_progress(self, task_id: str, completed_steps: int, status_message: str = "") -> None:
        """Update task progress.
        
        Args:
            task_id: Task identifier
            completed_steps: Number of completed steps
            status_message: Optional status message
        """
        with self.lock:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self.tasks[task_id]
            
            # Update task data - clamp completed_steps between 0 and total_steps
            previous_steps = task.completed_steps
            task.completed_steps = max(0, min(completed_steps, task.total_steps))
            task.status_message = status_message
            
            # Update Rich progress
            if task_id in self.rich_tasks:
                rich_task_id = self.rich_tasks[task_id]
                steps_advance = task.completed_steps - previous_steps
                
                self.progress.update(
                    rich_task_id,
                    advance=steps_advance,
                    status_message=status_message
                )
            
            # Calculate performance metrics
            elapsed_time = datetime.now() - task.start_time if task.start_time else timedelta(0)
            steps_per_second = task.completed_steps / elapsed_time.total_seconds() if elapsed_time.total_seconds() > 0 else 0
            
            if self.logger:
                self.logger.log_performance({
                    "task_id": task_id,
                    "completed_steps": task.completed_steps,
                    "total_steps": task.total_steps,
                    "progress_percentage": task.progress_percentage,
                    "elapsed_seconds": elapsed_time.total_seconds(),
                    "steps_per_second": steps_per_second,
                    "eta_seconds": (task.total_steps - task.completed_steps) / steps_per_second if steps_per_second > 0 else None
                })
    
    def complete_task(self, task_id: str, success: bool, summary: str) -> None:
        """Mark task as completed.
        
        Args:
            task_id: Task identifier
            success: Whether task completed successfully
            summary: Summary of task completion
        """
        with self.lock:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self.tasks[task_id]
            task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            task.end_time = datetime.now()
            task.status_message = summary
            
            # Complete Rich progress
            if task_id in self.rich_tasks:
                rich_task_id = self.rich_tasks[task_id]
                
                if success:
                    # Complete the progress bar
                    remaining_steps = task.total_steps - task.completed_steps
                    if remaining_steps > 0:
                        self.progress.update(rich_task_id, advance=remaining_steps)
                    
                    self.progress.update(
                        rich_task_id,
                        status_message=f"✅ {summary}"
                    )
                else:
                    self.progress.update(
                        rich_task_id,
                        status_message=f"❌ {summary}"
                    )
            
            # Calculate final metrics
            total_time = task.end_time - task.start_time if task.start_time and task.end_time else timedelta(0)
            
            if self.logger:
                self.logger.log_operation(
                    "task_completed",
                    {
                        "task_id": task_id,
                        "success": success,
                        "summary": summary,
                        "total_steps": task.total_steps,
                        "completed_steps": task.completed_steps,
                        "total_time_seconds": total_time.total_seconds(),
                        "end_time": task.end_time.isoformat()
                    }
                )
    
    def get_task_status(self, task_id: str) -> Optional[ProgressTask]:
        """Get current status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            ProgressTask object or None if task not found
        """
        with self.lock:
            return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, ProgressTask]:
        """Get all tracked tasks.
        
        Returns:
            Dictionary of all tasks
        """
        with self.lock:
            return self.tasks.copy()
    
    def get_active_tasks(self) -> Dict[str, ProgressTask]:
        """Get currently active tasks.
        
        Returns:
            Dictionary of active tasks
        """
        with self.lock:
            return {
                task_id: task 
                for task_id, task in self.tasks.items() 
                if task.status == TaskStatus.IN_PROGRESS
            }
    
    def get_summary_table(self) -> Table:
        """Generate a summary table of all tasks.
        
        Returns:
            Rich Table with task summary
        """
        table = Table(title="Task Summary")
        table.add_column("Task ID", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="green")
        table.add_column("Progress", style="yellow")
        table.add_column("Duration", style="blue")
        
        with self.lock:
            for task_id, task in self.tasks.items():
                status_emoji = {
                    TaskStatus.NOT_STARTED: "⏳",
                    TaskStatus.IN_PROGRESS: "🔄",
                    TaskStatus.COMPLETED: "✅",
                    TaskStatus.FAILED: "❌"
                }
                
                duration = ""
                if task.start_time:
                    end_time = task.end_time or datetime.now()
                    duration = str(end_time - task.start_time).split('.')[0]  # Remove microseconds
                
                table.add_row(
                    task_id,
                    task.description,
                    f"{status_emoji.get(task.status, '❓')} {task.status.value}",
                    f"{task.progress_percentage:.1f}% ({task.completed_steps}/{task.total_steps})",
                    duration
                )
        
        return table
    
    def clear_completed_tasks(self) -> None:
        """Remove completed and failed tasks from tracking."""
        with self.lock:
            completed_task_ids = [
                task_id for task_id, task in self.tasks.items()
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
            ]
            
            for task_id in completed_task_ids:
                # Remove from Rich progress
                if task_id in self.rich_tasks:
                    rich_task_id = self.rich_tasks[task_id]
                    self.progress.remove_task(rich_task_id)
                    del self.rich_tasks[task_id]
                
                # Remove from tasks
                del self.tasks[task_id]
            
            if self.logger:
                self.logger.log_operation(
                    "clear_completed_tasks",
                    {"cleared_count": len(completed_task_ids)}
                )
    
    def __enter__(self):
        """Context manager entry."""
        self.start_display()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_display()


def create_progress_tracker(logger: Optional[ILogger] = None, console: Optional[Console] = None) -> RichProgressTracker:
    """Factory function to create a progress tracker.
    
    Args:
        logger: Optional logger for integration
        console: Optional Rich console instance
        
    Returns:
        Configured RichProgressTracker instance
    """
    return RichProgressTracker(logger, console)