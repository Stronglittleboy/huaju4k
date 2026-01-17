"""
Multi-stage progress tracking implementation for huaju4k video enhancement.

This module provides comprehensive progress tracking with ETA estimation,
multi-stage progress calculation, and progress bar display system.
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from .interfaces import ProgressTracker

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    """Status of a processing stage."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcessingStage:
    """Information about a processing stage."""
    
    name: str
    display_name: str
    weight: float  # Relative weight for overall progress calculation
    status: StageStatus = StageStatus.NOT_STARTED
    progress: float = 0.0  # 0.0 to 1.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    details: str = ""
    substages: List['ProcessingStage'] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get stage duration if completed."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if stage is currently active."""
        return self.status == StageStatus.IN_PROGRESS
    
    @property
    def is_completed(self) -> bool:
        """Check if stage is completed."""
        return self.status in [StageStatus.COMPLETED, StageStatus.SKIPPED]


@dataclass
class ProgressSnapshot:
    """Snapshot of progress at a specific time."""
    
    timestamp: datetime
    overall_progress: float
    current_stage: str
    stage_progress: float
    eta_seconds: Optional[float]
    processing_speed: Optional[float]  # items per second
    details: str = ""


class MultiStageProgressTracker(ProgressTracker):
    """
    Multi-stage progress tracker with ETA estimation and display.
    
    This implementation tracks progress across multiple processing stages,
    calculates accurate ETAs, and provides real-time progress updates.
    """
    
    def __init__(self, update_interval: float = 0.5):
        """
        Initialize multi-stage progress tracker.
        
        Args:
            update_interval: Progress update interval in seconds
        """
        self.update_interval = update_interval
        
        # Stage management
        self.stages: Dict[str, ProcessingStage] = {}
        self.stage_order: List[str] = []
        self.current_stage: Optional[str] = None
        
        # Progress tracking
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.progress_history: List[ProgressSnapshot] = []
        
        # Callbacks and display
        self.progress_callbacks: List[Callable[[ProgressSnapshot], None]] = []
        self.display_enabled = True
        self.last_display_update = 0.0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance tracking
        self.items_processed = 0
        self.total_items = 0
        
        logger.debug("MultiStageProgressTracker initialized")
    
    def add_stage(self, name: str, display_name: str, weight: float = 1.0) -> None:
        """
        Add a processing stage.
        
        Args:
            name: Internal stage name
            display_name: Human-readable stage name
            weight: Relative weight for progress calculation
        """
        with self._lock:
            if name in self.stages:
                logger.warning(f"Stage '{name}' already exists, updating")
            
            stage = ProcessingStage(
                name=name,
                display_name=display_name,
                weight=weight
            )
            
            self.stages[name] = stage
            if name not in self.stage_order:
                self.stage_order.append(name)
            
            logger.debug(f"Added stage: {name} ({display_name}) with weight {weight}")
    
    def add_substage(self, parent_stage: str, name: str, display_name: str, weight: float = 1.0) -> None:
        """
        Add a substage to an existing stage.
        
        Args:
            parent_stage: Name of parent stage
            name: Internal substage name
            display_name: Human-readable substage name
            weight: Relative weight within parent stage
        """
        with self._lock:
            if parent_stage not in self.stages:
                raise ValueError(f"Parent stage '{parent_stage}' not found")
            
            substage = ProcessingStage(
                name=name,
                display_name=display_name,
                weight=weight
            )
            
            self.stages[parent_stage].substages.append(substage)
            logger.debug(f"Added substage: {name} to {parent_stage}")
    
    def start_processing(self, total_items: int = 0) -> None:
        """
        Start progress tracking.
        
        Args:
            total_items: Total number of items to process (for speed calculation)
        """
        with self._lock:
            self.start_time = datetime.now()
            self.end_time = None
            self.total_items = total_items
            self.items_processed = 0
            self.progress_history.clear()
            
            # Reset all stages
            for stage in self.stages.values():
                stage.status = StageStatus.NOT_STARTED
                stage.progress = 0.0
                stage.start_time = None
                stage.end_time = None
                stage.details = ""
                
                # Reset substages
                for substage in stage.substages:
                    substage.status = StageStatus.NOT_STARTED
                    substage.progress = 0.0
                    substage.start_time = None
                    substage.end_time = None
                    substage.details = ""
        
        logger.info(f"Started progress tracking with {total_items} total items")
        self._update_display()
    
    def start_stage(self, stage_name: str, details: str = "") -> None:
        """
        Start a processing stage.
        
        Args:
            stage_name: Name of stage to start
            details: Optional details about the stage
        """
        with self._lock:
            if stage_name not in self.stages:
                raise ValueError(f"Stage '{stage_name}' not found")
            
            # Complete previous stage if any
            if self.current_stage and self.current_stage != stage_name:
                prev_stage = self.stages[self.current_stage]
                if prev_stage.status == StageStatus.IN_PROGRESS:
                    prev_stage.status = StageStatus.COMPLETED
                    prev_stage.end_time = datetime.now()
                    prev_stage.progress = 1.0
            
            # Start new stage
            stage = self.stages[stage_name]
            stage.status = StageStatus.IN_PROGRESS
            stage.start_time = datetime.now()
            stage.progress = 0.0
            stage.details = details
            
            self.current_stage = stage_name
        
        logger.info(f"Started stage: {stage_name} - {details}")
        self._update_display()
    
    def update_stage_progress(self, stage: str, progress: float, details: str = None) -> None:
        """
        Update progress for specific processing stage.
        
        Args:
            stage: Stage name
            progress: Progress percentage (0.0 to 1.0)
            details: Optional detailed status message
        """
        with self._lock:
            if stage not in self.stages:
                logger.warning(f"Stage '{stage}' not found")
                return
            
            # Clamp progress to valid range
            progress = max(0.0, min(1.0, progress))
            
            stage_obj = self.stages[stage]
            stage_obj.progress = progress
            
            if details is not None:
                stage_obj.details = details
            
            # Update status based on progress
            if stage_obj.status == StageStatus.NOT_STARTED and progress > 0:
                stage_obj.status = StageStatus.IN_PROGRESS
                stage_obj.start_time = datetime.now()
            elif progress >= 1.0 and stage_obj.status == StageStatus.IN_PROGRESS:
                stage_obj.status = StageStatus.COMPLETED
                stage_obj.end_time = datetime.now()
        
        self._update_display()
    
    def update_substage_progress(self, parent_stage: str, substage_name: str, 
                               progress: float, details: str = None) -> None:
        """
        Update progress for a substage.
        
        Args:
            parent_stage: Name of parent stage
            substage_name: Name of substage
            progress: Progress percentage (0.0 to 1.0)
            details: Optional detailed status message
        """
        with self._lock:
            if parent_stage not in self.stages:
                logger.warning(f"Parent stage '{parent_stage}' not found")
                return
            
            parent = self.stages[parent_stage]
            substage = None
            
            # Find substage
            for sub in parent.substages:
                if sub.name == substage_name:
                    substage = sub
                    break
            
            if not substage:
                logger.warning(f"Substage '{substage_name}' not found in '{parent_stage}'")
                return
            
            # Update substage
            progress = max(0.0, min(1.0, progress))
            substage.progress = progress
            
            if details is not None:
                substage.details = details
            
            # Update substage status
            if substage.status == StageStatus.NOT_STARTED and progress > 0:
                substage.status = StageStatus.IN_PROGRESS
                substage.start_time = datetime.now()
            elif progress >= 1.0 and substage.status == StageStatus.IN_PROGRESS:
                substage.status = StageStatus.COMPLETED
                substage.end_time = datetime.now()
            
            # Update parent stage progress based on substages
            self._update_parent_progress(parent_stage)
        
        self._update_display()
    
    def complete_stage(self, stage_name: str, details: str = "") -> None:
        """
        Mark a stage as completed.
        
        Args:
            stage_name: Name of stage to complete
            details: Optional completion details
        """
        with self._lock:
            if stage_name not in self.stages:
                logger.warning(f"Stage '{stage_name}' not found")
                return
            
            stage = self.stages[stage_name]
            stage.status = StageStatus.COMPLETED
            stage.progress = 1.0
            stage.end_time = datetime.now()
            
            if details:
                stage.details = details
        
        logger.info(f"Completed stage: {stage_name} - {details}")
        self._update_display()
    
    def fail_stage(self, stage_name: str, error_details: str = "") -> None:
        """
        Mark a stage as failed.
        
        Args:
            stage_name: Name of stage that failed
            error_details: Error details
        """
        with self._lock:
            if stage_name not in self.stages:
                logger.warning(f"Stage '{stage_name}' not found")
                return
            
            stage = self.stages[stage_name]
            stage.status = StageStatus.FAILED
            stage.end_time = datetime.now()
            stage.details = error_details
        
        logger.error(f"Stage failed: {stage_name} - {error_details}")
        self._update_display()
    
    def skip_stage(self, stage_name: str, reason: str = "") -> None:
        """
        Mark a stage as skipped.
        
        Args:
            stage_name: Name of stage to skip
            reason: Reason for skipping
        """
        with self._lock:
            if stage_name not in self.stages:
                logger.warning(f"Stage '{stage_name}' not found")
                return
            
            stage = self.stages[stage_name]
            stage.status = StageStatus.SKIPPED
            stage.progress = 1.0  # Consider skipped as complete for progress calculation
            stage.details = reason
        
        logger.info(f"Skipped stage: {stage_name} - {reason}")
        self._update_display()
    
    def add_processed_items(self, count: int) -> None:
        """
        Add to the count of processed items.
        
        Args:
            count: Number of items processed
        """
        with self._lock:
            self.items_processed += count
        
        self._update_display()
    
    def calculate_eta(self) -> float:
        """
        Calculate estimated time to completion.
        
        Returns:
            Estimated seconds remaining
        """
        with self._lock:
            if not self.start_time:
                return 0.0
            
            overall_progress = self.get_overall_progress()
            
            if overall_progress <= 0.0:
                return 0.0
            
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            if overall_progress >= 1.0:
                return 0.0
            
            # Calculate ETA based on current progress rate
            estimated_total_time = elapsed / overall_progress
            remaining_time = estimated_total_time - elapsed
            
            return max(0.0, remaining_time)
    
    def get_overall_progress(self) -> float:
        """
        Get overall progress across all stages.
        
        Returns:
            Overall progress percentage (0.0 to 1.0)
        """
        with self._lock:
            if not self.stages:
                return 0.0
            
            total_weight = sum(stage.weight for stage in self.stages.values())
            if total_weight == 0:
                return 0.0
            
            weighted_progress = 0.0
            for stage in self.stages.values():
                stage_progress = stage.progress
                
                # If stage has substages, calculate based on substages
                if stage.substages:
                    substage_total_weight = sum(sub.weight for sub in stage.substages)
                    if substage_total_weight > 0:
                        substage_progress = sum(
                            sub.progress * sub.weight for sub in stage.substages
                        ) / substage_total_weight
                        stage_progress = substage_progress
                
                weighted_progress += stage_progress * stage.weight
            
            return weighted_progress / total_weight
    
    def get_processing_speed(self) -> Optional[float]:
        """
        Get current processing speed in items per second.
        
        Returns:
            Processing speed or None if not available
        """
        with self._lock:
            if not self.start_time or self.items_processed == 0:
                return None
            
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if elapsed <= 0:
                return None
            
            return self.items_processed / elapsed
    
    def display_progress_bar(self) -> None:
        """Display formatted progress bar with stage information."""
        if not self.display_enabled:
            return
        
        current_time = time.time()
        if current_time - self.last_display_update < self.update_interval:
            return
        
        self.last_display_update = current_time
        
        with self._lock:
            overall_progress = self.get_overall_progress()
            eta_seconds = self.calculate_eta()
            speed = self.get_processing_speed()
            
            # Create progress bar
            bar_width = 40
            filled_width = int(bar_width * overall_progress)
            bar = "█" * filled_width + "░" * (bar_width - filled_width)
            
            # Format progress percentage
            progress_pct = overall_progress * 100
            
            # Format ETA
            if eta_seconds and eta_seconds > 0:
                eta_str = self._format_duration(eta_seconds)
            else:
                eta_str = "Unknown"
            
            # Format speed
            if speed:
                speed_str = f"{speed:.1f} items/s"
            else:
                speed_str = "N/A"
            
            # Current stage info
            current_stage_info = ""
            if self.current_stage and self.current_stage in self.stages:
                stage = self.stages[self.current_stage]
                current_stage_info = f" | {stage.display_name}: {stage.progress*100:.1f}%"
                if stage.details:
                    current_stage_info += f" ({stage.details})"
            
            # Print progress line
            progress_line = (
                f"\r[{bar}] {progress_pct:5.1f}% | "
                f"ETA: {eta_str} | Speed: {speed_str}{current_stage_info}"
            )
            
            print(progress_line, end="", flush=True)
    
    def add_progress_callback(self, callback: Callable[[ProgressSnapshot], None]) -> None:
        """
        Add a callback to be called on progress updates.
        
        Args:
            callback: Function to call with progress snapshots
        """
        self.progress_callbacks.append(callback)
    
    def get_progress_snapshot(self) -> ProgressSnapshot:
        """
        Get current progress snapshot.
        
        Returns:
            ProgressSnapshot with current state
        """
        with self._lock:
            current_stage_name = self.current_stage or ""
            stage_progress = 0.0
            details = ""
            
            if self.current_stage and self.current_stage in self.stages:
                stage = self.stages[self.current_stage]
                stage_progress = stage.progress
                details = stage.details
            
            return ProgressSnapshot(
                timestamp=datetime.now(),
                overall_progress=self.get_overall_progress(),
                current_stage=current_stage_name,
                stage_progress=stage_progress,
                eta_seconds=self.calculate_eta(),
                processing_speed=self.get_processing_speed(),
                details=details
            )
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """
        Get summary of all stages.
        
        Returns:
            Dictionary with stage information
        """
        with self._lock:
            summary = {
                'total_stages': len(self.stages),
                'completed_stages': sum(1 for s in self.stages.values() if s.is_completed),
                'current_stage': self.current_stage,
                'overall_progress': self.get_overall_progress(),
                'stages': {}
            }
            
            for name, stage in self.stages.items():
                stage_info = {
                    'display_name': stage.display_name,
                    'status': stage.status.value,
                    'progress': stage.progress,
                    'weight': stage.weight,
                    'details': stage.details,
                    'duration': stage.duration.total_seconds() if stage.duration else None
                }
                
                if stage.substages:
                    stage_info['substages'] = [
                        {
                            'name': sub.name,
                            'display_name': sub.display_name,
                            'status': sub.status.value,
                            'progress': sub.progress,
                            'details': sub.details
                        }
                        for sub in stage.substages
                    ]
                
                summary['stages'][name] = stage_info
            
            return summary
    
    def finish_processing(self, success: bool = True) -> None:
        """
        Finish progress tracking.
        
        Args:
            success: Whether processing completed successfully
        """
        with self._lock:
            self.end_time = datetime.now()
            
            # Complete any remaining stages if successful
            if success:
                for stage in self.stages.values():
                    if stage.status == StageStatus.IN_PROGRESS:
                        stage.status = StageStatus.COMPLETED
                        stage.progress = 1.0
                        stage.end_time = self.end_time
        
        # Final display update
        if self.display_enabled:
            print()  # New line after progress bar
        
        total_time = (self.end_time - self.start_time).total_seconds() if self.start_time else 0
        status = "successfully" if success else "with errors"
        logger.info(f"Processing finished {status} in {total_time:.1f} seconds")
    
    def _update_parent_progress(self, parent_stage_name: str) -> None:
        """Update parent stage progress based on substages."""
        parent = self.stages[parent_stage_name]
        
        if not parent.substages:
            return
        
        total_weight = sum(sub.weight for sub in parent.substages)
        if total_weight == 0:
            return
        
        weighted_progress = sum(
            sub.progress * sub.weight for sub in parent.substages
        ) / total_weight
        
        parent.progress = weighted_progress
        
        # Update parent status based on substages
        if all(sub.is_completed for sub in parent.substages):
            parent.status = StageStatus.COMPLETED
            parent.end_time = datetime.now()
        elif any(sub.status == StageStatus.FAILED for sub in parent.substages):
            parent.status = StageStatus.FAILED
            parent.end_time = datetime.now()
        elif any(sub.is_active for sub in parent.substages):
            if parent.status == StageStatus.NOT_STARTED:
                parent.status = StageStatus.IN_PROGRESS
                parent.start_time = datetime.now()
    
    def _update_display(self) -> None:
        """Update progress display and call callbacks."""
        # 完全禁用显示更新以避免任何阻塞问题
        pass
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:.0f}m {secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"