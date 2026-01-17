"""
Core interfaces and abstract base classes for huaju4k video enhancement.

This module defines the main interfaces that all components must implement,
ensuring consistent behavior and enabling dependency injection.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from pathlib import Path

# Use TYPE_CHECKING to avoid runtime imports for type hints
if TYPE_CHECKING:
    import numpy as np

from ..models.data_models import (
    VideoInfo,
    ProcessingStrategy,
    TheaterFeatures,
    ProcessResult,
    TileConfiguration,
    CheckpointData,
    ResourceStatus,
    AudioResult
)


class VideoProcessor(ABC):
    """Abstract interface for video processing operations."""
    
    @abstractmethod
    def analyze_video(self, input_path: str) -> VideoInfo:
        """
        Analyze video properties and characteristics.
        
        Args:
            input_path: Path to input video file
            
        Returns:
            VideoInfo object containing video characteristics
        """
        pass
    
    @abstractmethod
    def process(self, input_path: str, output_path: str, 
                strategy: ProcessingStrategy) -> ProcessResult:
        """
        Process video with given strategy.
        
        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            strategy: Processing strategy configuration
            
        Returns:
            ProcessResult containing processing outcome
        """
        pass
    
    @abstractmethod
    def process_tiles(self, frames: List[Any], 
                     strategy: ProcessingStrategy) -> List[Any]:
        """
        Process video frames using tile-based approach.
        
        Args:
            frames: List of video frames as numpy arrays
            strategy: Processing strategy configuration
            
        Returns:
            List of processed frames
        """
        pass


class AudioEnhancer(ABC):
    """Abstract interface for theater audio enhancement."""
    
    @abstractmethod
    def analyze_theater_characteristics(self, audio_path: str) -> TheaterFeatures:
        """
        Analyze theater-specific audio characteristics.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            TheaterFeatures object containing analysis results
        """
        pass
    
    @abstractmethod
    def enhance(self, audio_path: str, output_path: str, 
                theater_preset: str = "medium") -> AudioResult:
        """
        Apply theater-specific audio enhancement.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path for output audio file
            theater_preset: Theater size preset (small, medium, large)
            
        Returns:
            AudioResult containing enhancement outcome
        """
        pass
    
    @abstractmethod
    def apply_venue_optimization(self, audio: Any, 
                                features: TheaterFeatures) -> Any:
        """
        Apply venue-specific audio processing.
        
        Args:
            audio: Audio data as numpy array
            features: Theater characteristics
            
        Returns:
            Processed audio data
        """
        pass


class MemoryManager(ABC):
    """Abstract interface for intelligent memory management."""
    
    @abstractmethod
    def calculate_optimal_tile_size(self, video_resolution: Tuple[int, int],
                                   model_memory_req: int) -> TileConfiguration:
        """
        Calculate optimal tile size based on available resources.
        
        Args:
            video_resolution: Input video resolution (width, height)
            model_memory_req: Memory requirement for AI model in MB
            
        Returns:
            TileConfiguration with optimal settings
        """
        pass
    
    @abstractmethod
    def monitor_and_adjust(self) -> ResourceStatus:
        """
        Monitor resource usage and adjust processing parameters.
        
        Returns:
            ResourceStatus containing current resource state
        """
        pass
    
    @abstractmethod
    def cleanup_resources(self) -> None:
        """Release all allocated resources and clean temporary files."""
        pass
    
    @abstractmethod
    def get_available_memory(self) -> Dict[str, int]:
        """
        Get available system and GPU memory.
        
        Returns:
            Dictionary with 'system' and 'gpu' memory in MB
        """
        pass


class ProgressTracker(ABC):
    """Abstract interface for multi-stage progress tracking."""
    
    @abstractmethod
    def update_stage_progress(self, stage: str, progress: float, 
                             details: str = None) -> None:
        """
        Update progress for specific processing stage.
        
        Args:
            stage: Stage name (analyzing, extracting, enhancing, etc.)
            progress: Progress percentage (0.0 to 1.0)
            details: Optional detailed status message
        """
        pass
    
    @abstractmethod
    def calculate_eta(self) -> float:
        """
        Calculate estimated time to completion.
        
        Returns:
            Estimated seconds remaining
        """
        pass
    
    @abstractmethod
    def display_progress_bar(self) -> None:
        """Display formatted progress bar with stage information."""
        pass
    
    @abstractmethod
    def get_overall_progress(self) -> float:
        """
        Get overall progress across all stages.
        
        Returns:
            Overall progress percentage (0.0 to 1.0)
        """
        pass


class ConfigurationManager(ABC):
    """Abstract interface for configuration and preset management."""
    
    @abstractmethod
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file or defaults.
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            Configuration dictionary
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid
        """
        pass
    
    @abstractmethod
    def load_preset(self, preset_name: str) -> Dict[str, Any]:
        """
        Load theater preset configuration.
        
        Args:
            preset_name: Name of preset (small, medium, large)
            
        Returns:
            Preset configuration dictionary
        """
        pass
    
    @abstractmethod
    def save_preset(self, preset_name: str, config: Dict[str, Any]) -> bool:
        """
        Save custom preset configuration.
        
        Args:
            preset_name: Name for the new preset
            config: Configuration to save
            
        Returns:
            True if preset was saved successfully
        """
        pass


class CheckpointSystem(ABC):
    """Abstract interface for checkpoint and recovery operations."""
    
    @abstractmethod
    def save_checkpoint(self, processor_state: Dict[str, Any], 
                       metadata: Dict[str, Any] = None) -> str:
        """
        Save processing state for recovery.
        
        Args:
            processor_state: Current processor state
            metadata: Optional metadata about the checkpoint
            
        Returns:
            Checkpoint identifier
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointData]:
        """
        Load checkpoint data for recovery.
        
        Args:
            checkpoint_id: Identifier of checkpoint to load
            
        Returns:
            CheckpointData if found, None otherwise
        """
        pass
    
    @abstractmethod
    def resume_processing(self, checkpoint_data: CheckpointData) -> bool:
        """
        Resume processing from checkpoint.
        
        Args:
            checkpoint_data: Checkpoint data to resume from
            
        Returns:
            True if resume was successful
        """
        pass
    
    @abstractmethod
    def cleanup_checkpoints(self, older_than_hours: int = 24) -> int:
        """
        Clean up old checkpoint files.
        
        Args:
            older_than_hours: Remove checkpoints older than this many hours
            
        Returns:
            Number of checkpoints removed
        """
        pass


class ErrorHandler(ABC):
    """Abstract interface for error handling and recovery."""
    
    @abstractmethod
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> str:
        """
        Handle processing error and determine recovery action.
        
        Args:
            error: Exception that occurred
            context: Processing context when error occurred
            
        Returns:
            Recovery action recommendation
        """
        pass
    
    @abstractmethod
    def generate_diagnostic_info(self, error: Exception, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed diagnostic information for error.
        
        Args:
            error: Exception that occurred
            context: Processing context when error occurred
            
        Returns:
            Diagnostic information dictionary
        """
        pass


class AIModelManager(ABC):
    """Abstract interface for AI model management."""
    
    @abstractmethod
    def load_model(self, model_name: str, use_gpu: bool = True) -> bool:
        """
        Load an AI model for use.
        
        Args:
            model_name: Name of the model to load
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            True if model loaded successfully
        """
        pass
    
    @abstractmethod
    def predict(self, image: Any) -> Any:
        """
        Perform prediction with current model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Upscaled image
        """
        pass
    
    @abstractmethod
    def predict_batch(self, images: List[Any]) -> List[Any]:
        """
        Perform batch prediction with current model.
        
        Args:
            images: List of input images
            
        Returns:
            List of upscaled images
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available model names.
        
        Returns:
            List of model names
        """
        pass
    
    @abstractmethod
    def unload_model(self, model_name: Optional[str] = None) -> None:
        """
        Unload a specific model or current model.
        
        Args:
            model_name: Name of model to unload, or None for current model
        """
        pass


class TileProcessor(ABC):
    """Abstract interface for tile-based processing."""
    
    @abstractmethod
    def process_image_tiles(self, image: Any, tile_config: TileConfiguration,
                           progress_callback: Optional[callable] = None) -> Any:
        """
        Process image using tile-based approach.
        
        Args:
            image: Input image as numpy array
            tile_config: Tile processing configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processed image as numpy array
        """
        pass
    
    @abstractmethod
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        pass