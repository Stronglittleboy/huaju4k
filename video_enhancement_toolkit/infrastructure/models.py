"""
Infrastructure Data Models

Data structures for infrastructure components.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TaskStatus(Enum):
    """Task status values."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    log_level: LogLevel = LogLevel.INFO
    log_file_path: Optional[str] = None
    console_output: bool = True
    json_format: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_file_size <= 0:
            raise ValueError("Max file size must be positive")
        if self.backup_count < 0:
            raise ValueError("Backup count must be non-negative")


@dataclass
class ProgressTask:
    """Progress tracking task data."""
    task_id: str
    description: str
    total_steps: int
    completed_steps: int = 0
    status: TaskStatus = TaskStatus.NOT_STARTED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status_message: str = ""
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100.0
    
    @property
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]


@dataclass
class ConfigurationData:
    """Complete configuration data structure."""
    video_config: Dict[str, Any] = field(default_factory=dict)
    audio_config: Dict[str, Any] = field(default_factory=dict)
    performance_config: Dict[str, Any] = field(default_factory=dict)
    logging_config: Dict[str, Any] = field(default_factory=dict)
    presets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def get_theater_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get built-in theater presets."""
        return {
            "small_theater": {
                "audio_config": {
                    "noise_reduction_strength": 0.4,
                    "dialogue_enhancement": 0.5,
                    "dynamic_range_target": -20.0,
                    "theater_preset": "small"
                },
                "video_config": {
                    "tile_size": 512,
                    "batch_size": 4,
                    "ai_model": "realesrgan-x4plus"
                }
            },
            "medium_theater": {
                "audio_config": {
                    "noise_reduction_strength": 0.3,
                    "dialogue_enhancement": 0.4,
                    "dynamic_range_target": -23.0,
                    "theater_preset": "medium"
                },
                "video_config": {
                    "tile_size": 400,
                    "batch_size": 6,
                    "ai_model": "realesrgan-x4plus"
                }
            },
            "large_theater": {
                "audio_config": {
                    "noise_reduction_strength": 0.2,
                    "dialogue_enhancement": 0.3,
                    "dynamic_range_target": -26.0,
                    "theater_preset": "large"
                },
                "video_config": {
                    "tile_size": 256,
                    "batch_size": 8,
                    "ai_model": "real-cugan"
                }
            }
        }