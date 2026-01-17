"""
Core Processing Interfaces

Abstract base classes and interfaces for all major processing components.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np

from .models import VideoConfig, AudioConfig, PerformanceConfig, AudioAnalysis, QualityMetrics


class IVideoProcessor(ABC):
    """Interface for video processing operations."""
    
    @abstractmethod
    def extract_frames(self, video_path: str, output_dir: str) -> List[str]:
        """Extract video frames for processing.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save extracted frames
            
        Returns:
            List of paths to extracted frame files
        """
        pass
    
    @abstractmethod
    def enhance_frames(self, frame_paths: List[str]) -> List[str]:
        """Apply AI enhancement to frames.
        
        Args:
            frame_paths: List of paths to frame files
            
        Returns:
            List of paths to enhanced frame files
        """
        pass
    
    @abstractmethod
    def reassemble_video(self, enhanced_frames: List[str], output_path: str) -> str:
        """Reassemble enhanced frames into video.
        
        Args:
            enhanced_frames: List of paths to enhanced frame files
            output_path: Path for output video file
            
        Returns:
            Path to final enhanced video file
        """
        pass
    
    @abstractmethod
    def extract_audio(self, video_path: str, output_path: str) -> str:
        """Extract audio track from video file.
        
        Args:
            video_path: Path to input video file
            output_path: Path for extracted audio file
            
        Returns:
            Path to extracted audio file, empty string if failed
        """
        pass
    
    @abstractmethod
    def merge_audio_video(self, video_path: str, audio_path: str, output_path: str) -> str:
        """Merge audio and video files with synchronization preservation.
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            output_path: Path for merged output file
            
        Returns:
            Path to merged video file, empty string if failed
        """
        pass
    
    @abstractmethod
    def validate_processing_stage(self, stage_name: str, input_files: List[str], output_files: List[str]) -> Dict[str, Any]:
        """Validate intermediate processing stage results.
        
        Args:
            stage_name: Name of the processing stage
            input_files: List of input file paths
            output_files: List of output file paths
            
        Returns:
            Dictionary with validation results
        """
        pass
    
    @abstractmethod
    def validate_final_output(self, output_path: str, original_video_path: str) -> Dict[str, Any]:
        """Validate final enhanced video output quality and properties.
        
        Args:
            output_path: Path to final enhanced video
            original_video_path: Path to original video for comparison
            
        Returns:
            Dictionary with validation results
        """
        pass


class IAudioOptimizer(ABC):
    """Interface for theater audio optimization."""
    
    @abstractmethod
    def analyze_audio_characteristics(self, audio_path: str) -> AudioAnalysis:
        """Analyze theater-specific audio characteristics.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            AudioAnalysis with detected characteristics
        """
        pass
    
    @abstractmethod
    def apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply intelligent noise reduction without over-processing.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Processed audio data with reduced noise
        """
        pass
    
    @abstractmethod
    def optimize_dialogue_clarity(self, audio_data: np.ndarray) -> np.ndarray:
        """Enhance dialogue frequencies while preserving naturalness.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Audio data with enhanced dialogue clarity
        """
        pass
    
    @abstractmethod
    def validate_audio_quality(self, original: np.ndarray, processed: np.ndarray) -> QualityMetrics:
        """Validate processed audio to prevent distortion.
        
        Args:
            original: Original audio data
            processed: Processed audio data
            
        Returns:
            QualityMetrics with validation results
        """
        pass


class IPerformanceManager(ABC):
    """Interface for performance management and hardware optimization."""
    
    @abstractmethod
    def analyze_system_resources(self) -> dict:
        """Analyze available hardware resources.
        
        Returns:
            Dictionary with system resource information
        """
        pass
    
    @abstractmethod
    def optimize_processing_parameters(self, task_type: str) -> PerformanceConfig:
        """Optimize parameters based on available hardware.
        
        Args:
            task_type: Type of processing task
            
        Returns:
            PerformanceConfig with optimized parameters
        """
        pass
    
    @abstractmethod
    def allocate_resources(self, cpu_threads: int, gpu_usage: bool) -> dict:
        """Allocate optimal system resources.
        
        Args:
            cpu_threads: Number of CPU threads to use
            gpu_usage: Whether to use GPU acceleration
            
        Returns:
            Dictionary with resource allocation details
        """
        pass
    
    @abstractmethod
    def monitor_performance(self) -> dict:
        """Monitor real-time performance metrics.
        
        Returns:
            Dictionary with current performance metrics
        """
        pass