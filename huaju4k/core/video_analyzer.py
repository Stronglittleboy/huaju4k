"""
Video Analysis System for huaju4k Video Enhancement.

This module provides comprehensive video analysis capabilities including
property detection, processing strategy calculation, and tile configuration
optimization for theater drama content.
"""

import os
import subprocess
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

from ..models.data_models import (
    VideoInfo, ProcessingStrategy, TileConfiguration, 
    VideoConfig, PerformanceConfig
)
from .interfaces import VideoProcessor


class VideoAnalyzer:
    """
    Analyzes video files to extract properties and determine optimal processing strategies.
    
    This class implements video property detection, processing strategy calculation,
    and tile configuration optimization as required by task 6.1.
    """
    
    def __init__(self, config: Optional[VideoConfig] = None, 
                 performance_config: Optional[PerformanceConfig] = None):
        """
        Initialize the video analyzer.
        
        Args:
            config: Video processing configuration
            performance_config: Performance and resource configuration
        """
        self.config = config or VideoConfig()
        self.performance_config = performance_config or PerformanceConfig()
        self.logger = logging.getLogger(__name__)
        
    def analyze_video(self, input_path: str) -> VideoInfo:
        """
        Analyze video properties and characteristics.
        
        Implements comprehensive video analysis including resolution, codec,
        framerate detection as required by Requirements 3.1.
        
        Args:
            input_path: Path to input video file
            
        Returns:
            VideoInfo object containing video characteristics
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If analysis fails
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Video file not found: {input_path}")
            
        self.logger.info(f"Starting video analysis: {input_path}")
        
        try:
            # Extract metadata using FFprobe
            metadata = self._extract_metadata_ffprobe(input_path)
            
            # Get file size
            file_size = os.path.getsize(input_path)
            
            # Create VideoInfo object
            video_info = VideoInfo(
                resolution=metadata['resolution'],
                duration=metadata['duration'],
                framerate=metadata['framerate'],
                codec=metadata['codec'],
                bitrate=metadata['bitrate'],
                has_audio=metadata['has_audio'],
                audio_channels=metadata['audio_channels'],
                audio_sample_rate=metadata['audio_sample_rate'],
                file_size=file_size,
                format=metadata['format']
            )
            
            self.logger.info(f"Video analysis completed: {video_info.resolution[0]}x{video_info.resolution[1]}, "
                           f"{video_info.duration:.1f}s, {video_info.framerate:.1f}fps")
            
            return video_info
            
        except Exception as e:
            self.logger.error(f"Video analysis failed: {str(e)}")
            raise RuntimeError(f"Failed to analyze video: {str(e)}")
    
    def calculate_processing_strategy(self, video_info: VideoInfo, 
                                    quality_level: str = "balanced",
                                    target_resolution: Optional[Tuple[int, int]] = None) -> ProcessingStrategy:
        """
        Calculate optimal processing strategy based on video characteristics.
        
        Implements processing strategy calculation based on video characteristics
        as required by Requirements 3.1.
        
        Args:
            video_info: Video information from analysis
            quality_level: Quality preset (fast, balanced, high)
            target_resolution: Desired output resolution (defaults to 4K)
            
        Returns:
            ProcessingStrategy with optimized parameters
        """
        self.logger.info(f"Calculating processing strategy for {quality_level} quality")
        
        # Get quality preset parameters
        preset_params = self.config.get_preset(quality_level)
        
        # Determine target resolution (default to 4K if not specified)
        if target_resolution is None:
            # Calculate 4K target based on aspect ratio
            aspect_ratio = video_info.aspect_ratio
            if aspect_ratio >= 16/9 - 0.1:  # 16:9 or wider
                target_resolution = (3840, 2160)
            else:  # 4:3 or other ratios
                target_resolution = (2880, 2160)  # Maintain height, adjust width
        
        # Calculate optimal tile size based on available memory
        tile_config = self._calculate_optimal_tile_size(
            video_info.resolution, 
            preset_params['tile_size'],
            quality_level
        )
        
        # Determine GPU usage based on availability and video characteristics
        use_gpu = self._should_use_gpu(video_info, quality_level)
        
        # Calculate memory limit
        memory_limit_mb = self._calculate_memory_limit(video_info, tile_config)
        
        strategy = ProcessingStrategy(
            tile_size=(tile_config.tile_width, tile_config.tile_height),
            overlap_pixels=tile_config.overlap,
            batch_size=tile_config.batch_size,
            use_gpu=use_gpu,
            ai_model=self.config.ai_model,
            quality_preset=quality_level,
            memory_limit_mb=memory_limit_mb,
            target_resolution=target_resolution,
            denoise_strength=preset_params['denoise_strength']
        )
        
        self.logger.info(f"Processing strategy calculated: {strategy.tile_size} tiles, "
                        f"batch_size={strategy.batch_size}, gpu={strategy.use_gpu}")
        
        return strategy
    
    def optimize_tile_configuration(self, video_resolution: Tuple[int, int],
                                  available_memory_mb: int,
                                  quality_level: str = "balanced") -> TileConfiguration:
        """
        Create optimized tile configuration for memory-efficient processing.
        
        Implements tile configuration optimization as required by Requirements 3.1.
        
        Args:
            video_resolution: Input video resolution (width, height)
            available_memory_mb: Available system memory in MB
            quality_level: Quality preset affecting tile size
            
        Returns:
            TileConfiguration with optimal settings
        """
        self.logger.info(f"Optimizing tile configuration for {video_resolution} resolution")
        
        # Get base tile size from quality preset
        preset_params = self.config.get_preset(quality_level)
        base_tile_size = preset_params['tile_size']
        
        # Calculate optimal tile dimensions
        tile_width, tile_height = self._calculate_tile_dimensions(
            video_resolution, base_tile_size, available_memory_mb
        )
        
        # Calculate overlap based on tile size (larger tiles need more overlap)
        overlap = max(32, min(64, tile_width // 16))
        
        # Calculate batch size based on memory constraints
        batch_size = self._calculate_batch_size(
            tile_width, tile_height, available_memory_mb
        )
        
        # Estimate memory usage
        memory_usage_mb = self._estimate_tile_memory_usage(
            tile_width, tile_height, batch_size
        )
        
        # Determine processing mode
        processing_mode = "gpu" if self.performance_config.use_gpu else "cpu"
        
        tile_config = TileConfiguration(
            tile_width=tile_width,
            tile_height=tile_height,
            overlap=overlap,
            batch_size=batch_size,
            memory_usage_mb=memory_usage_mb,
            processing_mode=processing_mode
        )
        
        self.logger.info(f"Tile configuration optimized: {tile_width}x{tile_height}, "
                        f"batch_size={batch_size}, memory={memory_usage_mb}MB")
        
        return tile_config
    
    def _extract_metadata_ffprobe(self, video_path: str) -> Dict[str, Any]:
        """
        Extract video metadata using FFprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing metadata
        """
        try:
            # Run ffprobe to get detailed video information
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)
            
            # Extract video stream information
            video_stream = None
            audio_streams = []
            
            for stream in probe_data['streams']:
                if stream['codec_type'] == 'video' and video_stream is None:
                    video_stream = stream
                elif stream['codec_type'] == 'audio':
                    audio_streams.append(stream)
            
            if not video_stream:
                raise RuntimeError("No video stream found in file")
            
            # Parse video metadata
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            
            # Parse frame rate
            frame_rate_str = video_stream.get('r_frame_rate', '25/1')
            if '/' in frame_rate_str:
                num, den = map(int, frame_rate_str.split('/'))
                framerate = num / den if den != 0 else 25.0
            else:
                framerate = float(frame_rate_str)
            
            # Parse duration
            duration = float(video_stream.get('duration', 
                           probe_data.get('format', {}).get('duration', 0)))
            
            # Parse bitrate
            bitrate = int(video_stream.get('bit_rate', 
                         probe_data.get('format', {}).get('bit_rate', 0)))
            
            # Audio information
            has_audio = len(audio_streams) > 0
            audio_channels = 0
            audio_sample_rate = 0
            
            if has_audio:
                # Use first audio stream for basic info
                audio_stream = audio_streams[0]
                audio_channels = int(audio_stream.get('channels', 2))
                audio_sample_rate = int(audio_stream.get('sample_rate', 44100))
            
            # Container format
            format_name = probe_data.get('format', {}).get('format_name', 'unknown')
            
            return {
                'resolution': (width, height),
                'framerate': framerate,
                'duration': duration,
                'codec': video_stream.get('codec_name', 'unknown'),
                'bitrate': bitrate,
                'has_audio': has_audio,
                'audio_channels': audio_channels,
                'audio_sample_rate': audio_sample_rate,
                'format': format_name
            }
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFprobe failed: {e.stderr}")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise RuntimeError(f"Failed to parse FFprobe output: {str(e)}")
    
    def _calculate_optimal_tile_size(self, video_resolution: Tuple[int, int],
                                   base_tile_size: int,
                                   quality_level: str) -> TileConfiguration:
        """Calculate optimal tile size based on video resolution and available memory."""
        width, height = video_resolution
        
        # Adjust base tile size based on input resolution
        if width >= 1920:  # HD or higher
            tile_size_factor = 1.0
        elif width >= 1280:  # Lower HD
            tile_size_factor = 0.8
        else:  # SD
            tile_size_factor = 0.6
        
        # Apply quality adjustments
        quality_factors = {
            "fast": 0.8,
            "balanced": 1.0,
            "high": 1.2
        }
        
        adjusted_tile_size = int(base_tile_size * tile_size_factor * 
                               quality_factors.get(quality_level, 1.0))
        
        # Ensure tile size is reasonable and divisible by 8 (for video encoding)
        adjusted_tile_size = max(256, min(1024, adjusted_tile_size))
        adjusted_tile_size = (adjusted_tile_size // 8) * 8
        
        # Get available memory (simplified - would use actual memory manager)
        available_memory_mb = 4000  # Default assumption, would be dynamic
        
        return self.optimize_tile_configuration(
            video_resolution, available_memory_mb, quality_level
        )
    
    def _calculate_tile_dimensions(self, video_resolution: Tuple[int, int],
                                 base_tile_size: int,
                                 available_memory_mb: int) -> Tuple[int, int]:
        """Calculate optimal tile width and height."""
        width, height = video_resolution
        
        # Start with square tiles
        tile_width = tile_height = base_tile_size
        
        # Adjust for very wide or tall videos
        aspect_ratio = width / height
        if aspect_ratio > 2.0:  # Very wide
            tile_width = int(tile_width * 1.2)
            tile_height = int(tile_height * 0.8)
        elif aspect_ratio < 0.8:  # Very tall
            tile_width = int(tile_width * 0.8)
            tile_height = int(tile_height * 1.2)
        
        # Ensure dimensions are reasonable and divisible by 8
        tile_width = max(256, min(1024, (tile_width // 8) * 8))
        tile_height = max(256, min(1024, (tile_height // 8) * 8))
        
        return tile_width, tile_height
    
    def _calculate_batch_size(self, tile_width: int, tile_height: int,
                            available_memory_mb: int) -> int:
        """Calculate optimal batch size based on tile size and available memory."""
        # Estimate memory per tile (rough calculation)
        # Each tile uses memory for: input (3 channels) + output (3 channels) + model overhead
        bytes_per_pixel = 4  # Float32
        tile_memory_mb = (tile_width * tile_height * 3 * 2 * bytes_per_pixel) / (1024 * 1024)
        
        # Add model overhead (estimated)
        model_overhead_mb = 500  # Rough estimate for AI model
        
        # Calculate safe batch size
        available_for_tiles = available_memory_mb * self.performance_config.max_memory_usage - model_overhead_mb
        max_batch_size = max(1, int(available_for_tiles / tile_memory_mb))
        
        # Limit batch size to reasonable values
        return min(max_batch_size, 8)
    
    def _estimate_tile_memory_usage(self, tile_width: int, tile_height: int,
                                  batch_size: int) -> int:
        """Estimate memory usage for tile processing."""
        bytes_per_pixel = 4  # Float32
        tile_memory_mb = (tile_width * tile_height * 3 * 2 * bytes_per_pixel) / (1024 * 1024)
        
        # Total memory = (tiles * batch_size) + model overhead
        total_memory_mb = int(tile_memory_mb * batch_size + 500)  # 500MB model overhead
        
        return total_memory_mb
    
    def _should_use_gpu(self, video_info: VideoInfo, quality_level: str) -> bool:
        """Determine if GPU should be used based on video characteristics."""
        if not self.performance_config.use_gpu:
            return False
        
        # Check if video is large enough to benefit from GPU
        total_pixels = video_info.resolution[0] * video_info.resolution[1]
        
        # GPU is beneficial for larger videos and higher quality settings
        if total_pixels >= 1920 * 1080 and quality_level in ["balanced", "high"]:
            return True
        elif total_pixels >= 3840 * 2160:  # Always use GPU for 4K+ input
            return True
        
        return False
    
    def _calculate_memory_limit(self, video_info: VideoInfo, 
                              tile_config: TileConfiguration) -> int:
        """Calculate memory limit for processing."""
        # Base memory limit from configuration
        base_limit = int(4000 * self.performance_config.max_memory_usage)  # 4GB default
        
        # Adjust based on video size
        total_pixels = video_info.resolution[0] * video_info.resolution[1]
        if total_pixels >= 3840 * 2160:  # 4K+
            base_limit = int(base_limit * 1.5)
        elif total_pixels <= 1280 * 720:  # HD or lower
            base_limit = int(base_limit * 0.7)
        
        # Ensure we have enough for tile processing
        min_required = tile_config.memory_usage_mb + 1000  # 1GB buffer
        
        return max(base_limit, min_required)