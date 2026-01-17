"""
Processing Strategy Optimizer for huaju4k Video Enhancement.

This module implements quality level parameter adjustment and adaptive 
processing parameter selection as required by Task 6.3 and Requirements 2.4.
"""

from typing import Dict, Any, Tuple, Optional
import logging

from ..models.data_models import (
    VideoInfo, ProcessingStrategy, VideoConfig, 
    PerformanceConfig, TileConfiguration
)


class ProcessingStrategyOptimizer:
    """
    Optimizes processing strategies based on quality levels and video characteristics.
    
    Implements Task 6.3 requirements:
    - Add quality level parameter adjustment
    - Create adaptive processing parameter selection
    - Requirements: 2.4
    """
    
    def __init__(self, video_config: Optional[VideoConfig] = None,
                 performance_config: Optional[PerformanceConfig] = None):
        """
        Initialize the strategy optimizer.
        
        Args:
            video_config: Video processing configuration
            performance_config: Performance and resource configuration
        """
        self.video_config = video_config or VideoConfig()
        self.performance_config = performance_config or PerformanceConfig()
        self.logger = logging.getLogger(__name__)
    
    def adjust_quality_parameters(self, base_strategy: ProcessingStrategy,
                                quality_level: str) -> ProcessingStrategy:
        """
        Adjust processing parameters based on quality level.
        
        Implements quality level parameter adjustment as required by Requirements 2.4.
        
        Args:
            base_strategy: Base processing strategy
            quality_level: Quality preset (fast, balanced, high)
            
        Returns:
            ProcessingStrategy with adjusted parameters
        """
        self.logger.info(f"Adjusting parameters for quality level: {quality_level}")
        
        # Get quality preset parameters
        preset_params = self.video_config.get_preset(quality_level)
        
        # Quality-specific adjustments
        quality_adjustments = self._get_quality_adjustments(quality_level)
        
        # Create optimized strategy
        optimized_strategy = ProcessingStrategy(
            tile_size=self._adjust_tile_size(base_strategy.tile_size, quality_adjustments),
            overlap_pixels=self._adjust_overlap(base_strategy.overlap_pixels, quality_adjustments),
            batch_size=self._adjust_batch_size(base_strategy.batch_size, quality_adjustments),
            use_gpu=base_strategy.use_gpu,
            ai_model=base_strategy.ai_model,
            quality_preset=quality_level,
            memory_limit_mb=base_strategy.memory_limit_mb,
            target_resolution=base_strategy.target_resolution,
            denoise_strength=preset_params['denoise_strength']
        )
        
        self.logger.info(f"Quality parameters adjusted: tile_size={optimized_strategy.tile_size}, "
                        f"denoise_strength={optimized_strategy.denoise_strength}")
        
        return optimized_strategy
    
    def select_adaptive_parameters(self, video_info: VideoInfo,
                                 quality_level: str,
                                 available_memory_mb: int) -> ProcessingStrategy:
        """
        Create adaptive processing parameter selection based on video characteristics.
        
        Implements adaptive processing parameter selection as required by Requirements 2.4.
        
        Args:
            video_info: Video information and characteristics
            quality_level: Desired quality level
            available_memory_mb: Available system memory
            
        Returns:
            ProcessingStrategy with adaptively selected parameters
        """
        self.logger.info(f"Selecting adaptive parameters for {video_info.resolution} video")
        
        # Analyze video characteristics for adaptive selection
        video_complexity = self._analyze_video_complexity(video_info)
        processing_requirements = self._calculate_processing_requirements(video_info, quality_level)
        
        # Adaptive parameter selection
        tile_size = self._select_adaptive_tile_size(video_info, quality_level, available_memory_mb)
        overlap_pixels = self._select_adaptive_overlap(tile_size, video_complexity)
        batch_size = self._select_adaptive_batch_size(tile_size, available_memory_mb, quality_level)
        memory_limit = self._select_adaptive_memory_limit(video_info, available_memory_mb)
        
        # GPU usage decision
        use_gpu = self._should_use_gpu_adaptive(video_info, quality_level, processing_requirements)
        
        # Target resolution adaptation
        target_resolution = self._select_adaptive_target_resolution(video_info)
        
        # Get quality preset parameters
        preset_params = self.video_config.get_preset(quality_level)
        
        strategy = ProcessingStrategy(
            tile_size=tile_size,
            overlap_pixels=overlap_pixels,
            batch_size=batch_size,
            use_gpu=use_gpu,
            ai_model=self.video_config.ai_model,
            quality_preset=quality_level,
            memory_limit_mb=memory_limit,
            target_resolution=target_resolution,
            denoise_strength=preset_params['denoise_strength']
        )
        
        self.logger.info(f"Adaptive parameters selected: complexity={video_complexity}, "
                        f"gpu={use_gpu}, tile_size={tile_size}")
        
        return strategy
    
    def _get_quality_adjustments(self, quality_level: str) -> Dict[str, float]:
        """Get quality-specific adjustment factors."""
        adjustments = {
            "fast": {
                "tile_size_factor": 0.8,
                "overlap_factor": 0.8,
                "batch_size_factor": 1.5,
                "memory_factor": 0.8
            },
            "balanced": {
                "tile_size_factor": 1.0,
                "overlap_factor": 1.0,
                "batch_size_factor": 1.0,
                "memory_factor": 1.0
            },
            "high": {
                "tile_size_factor": 1.2,
                "overlap_factor": 1.2,
                "batch_size_factor": 0.8,
                "memory_factor": 1.3
            }
        }
        
        return adjustments.get(quality_level, adjustments["balanced"])
    
    def _adjust_tile_size(self, base_tile_size: Tuple[int, int], 
                         adjustments: Dict[str, float]) -> Tuple[int, int]:
        """Adjust tile size based on quality level."""
        factor = adjustments["tile_size_factor"]
        width, height = base_tile_size
        
        new_width = int(width * factor)
        new_height = int(height * factor)
        
        # Ensure dimensions are reasonable and divisible by 8
        new_width = max(256, min(1024, (new_width // 8) * 8))
        new_height = max(256, min(1024, (new_height // 8) * 8))
        
        return (new_width, new_height)
    
    def _adjust_overlap(self, base_overlap: int, adjustments: Dict[str, float]) -> int:
        """Adjust overlap pixels based on quality level."""
        factor = adjustments["overlap_factor"]
        new_overlap = int(base_overlap * factor)
        
        # Ensure overlap is reasonable
        return max(16, min(128, new_overlap))
    
    def _adjust_batch_size(self, base_batch_size: int, adjustments: Dict[str, float]) -> int:
        """Adjust batch size based on quality level."""
        factor = adjustments["batch_size_factor"]
        new_batch_size = int(base_batch_size * factor)
        
        # Ensure batch size is reasonable
        return max(1, min(16, new_batch_size))
    
    def _analyze_video_complexity(self, video_info: VideoInfo) -> str:
        """Analyze video complexity for adaptive parameter selection."""
        total_pixels = video_info.resolution[0] * video_info.resolution[1]
        
        # Complexity based on resolution and duration
        if total_pixels >= 3840 * 2160:  # 4K+
            return "high"
        elif total_pixels >= 1920 * 1080:  # HD
            if video_info.duration > 3600:  # > 1 hour
                return "high"
            elif video_info.duration > 1800:  # > 30 minutes
                return "medium"
            else:
                return "medium"
        else:  # SD
            return "low"
    
    def _calculate_processing_requirements(self, video_info: VideoInfo, 
                                         quality_level: str) -> Dict[str, Any]:
        """Calculate processing requirements based on video and quality."""
        total_pixels = video_info.resolution[0] * video_info.resolution[1]
        total_frames = video_info.total_frames
        
        # Estimate processing load
        pixel_load = total_pixels * total_frames
        
        # Quality multipliers
        quality_multipliers = {
            "fast": 1.0,
            "balanced": 1.5,
            "high": 2.0
        }
        
        processing_load = pixel_load * quality_multipliers.get(quality_level, 1.0)
        
        return {
            "processing_load": processing_load,
            "estimated_time_hours": processing_load / (1920 * 1080 * 30 * 3600),  # Rough estimate
            "memory_intensive": total_pixels >= 1920 * 1080,
            "gpu_beneficial": total_pixels >= 1920 * 1080 and quality_level != "fast"
        }
    
    def _select_adaptive_tile_size(self, video_info: VideoInfo, quality_level: str,
                                 available_memory_mb: int) -> Tuple[int, int]:
        """Select adaptive tile size based on video characteristics."""
        width, height = video_info.resolution
        
        # Base tile size from quality preset
        preset_params = self.video_config.get_preset(quality_level)
        base_size = preset_params['tile_size']
        
        # Adapt based on video resolution
        if width >= 3840:  # 4K+
            tile_factor = 1.2
        elif width >= 1920:  # HD
            tile_factor = 1.0
        else:  # SD
            tile_factor = 0.8
        
        # Adapt based on available memory
        if available_memory_mb < 4000:  # < 4GB
            memory_factor = 0.8
        elif available_memory_mb > 8000:  # > 8GB
            memory_factor = 1.2
        else:
            memory_factor = 1.0
        
        # Calculate adaptive tile size
        adaptive_size = int(base_size * tile_factor * memory_factor)
        
        # Ensure reasonable bounds and divisibility by 8
        adaptive_size = max(256, min(1024, (adaptive_size // 8) * 8))
        
        # For very wide videos, adjust aspect ratio
        aspect_ratio = width / height
        if aspect_ratio > 2.0:  # Very wide
            tile_width = int(adaptive_size * 1.2)
            tile_height = int(adaptive_size * 0.8)
        elif aspect_ratio < 0.8:  # Very tall
            tile_width = int(adaptive_size * 0.8)
            tile_height = int(adaptive_size * 1.2)
        else:
            tile_width = tile_height = adaptive_size
        
        # Final bounds check
        tile_width = max(256, min(1024, (tile_width // 8) * 8))
        tile_height = max(256, min(1024, (tile_height // 8) * 8))
        
        return (tile_width, tile_height)
    
    def _select_adaptive_overlap(self, tile_size: Tuple[int, int], complexity: str) -> int:
        """Select adaptive overlap based on tile size and complexity."""
        avg_tile_size = (tile_size[0] + tile_size[1]) // 2
        
        # Base overlap as fraction of tile size
        base_overlap = avg_tile_size // 16
        
        # Adjust based on complexity
        complexity_factors = {
            "low": 0.8,
            "medium": 1.0,
            "high": 1.2
        }
        
        adaptive_overlap = int(base_overlap * complexity_factors.get(complexity, 1.0))
        
        # Ensure reasonable bounds
        return max(16, min(128, adaptive_overlap))
    
    def _select_adaptive_batch_size(self, tile_size: Tuple[int, int],
                                  available_memory_mb: int, quality_level: str) -> int:
        """Select adaptive batch size based on tile size and memory."""
        tile_width, tile_height = tile_size
        
        # Estimate memory per tile (rough calculation)
        bytes_per_pixel = 4  # Float32
        tile_memory_mb = (tile_width * tile_height * 3 * 2 * bytes_per_pixel) / (1024 * 1024)
        
        # Available memory for tiles (reserve memory for model)
        model_memory_mb = 1000  # Reserve 1GB for model
        available_for_tiles = (available_memory_mb * self.performance_config.max_memory_usage 
                             - model_memory_mb)
        
        # Calculate max batch size
        max_batch_size = max(1, int(available_for_tiles / tile_memory_mb))
        
        # Quality-based adjustment
        quality_batch_factors = {
            "fast": 1.5,
            "balanced": 1.0,
            "high": 0.8
        }
        
        adaptive_batch_size = int(max_batch_size * quality_batch_factors.get(quality_level, 1.0))
        
        # Ensure reasonable bounds
        return max(1, min(8, adaptive_batch_size))
    
    def _select_adaptive_memory_limit(self, video_info: VideoInfo, 
                                    available_memory_mb: int) -> int:
        """Select adaptive memory limit based on video characteristics."""
        total_pixels = video_info.resolution[0] * video_info.resolution[1]
        
        # Base memory limit
        base_limit = int(available_memory_mb * self.performance_config.max_memory_usage)
        
        # Adjust based on video size
        if total_pixels >= 3840 * 2160:  # 4K+
            memory_factor = 1.3
        elif total_pixels >= 1920 * 1080:  # HD
            memory_factor = 1.0
        else:  # SD
            memory_factor = 0.7
        
        adaptive_limit = int(base_limit * memory_factor)
        
        # Ensure minimum memory for processing
        return max(2000, adaptive_limit)  # At least 2GB
    
    def _should_use_gpu_adaptive(self, video_info: VideoInfo, quality_level: str,
                               processing_requirements: Dict[str, Any]) -> bool:
        """Adaptively decide whether to use GPU based on video characteristics."""
        if not self.performance_config.use_gpu:
            return False
        
        # GPU beneficial for larger videos and higher quality
        if processing_requirements["gpu_beneficial"]:
            return True
        
        # For smaller videos, only use GPU for high quality
        total_pixels = video_info.resolution[0] * video_info.resolution[1]
        if total_pixels < 1920 * 1080 and quality_level == "fast":
            return False
        
        return True
    
    def _select_adaptive_target_resolution(self, video_info: VideoInfo) -> Tuple[int, int]:
        """Select adaptive target resolution based on input video."""
        width, height = video_info.resolution
        aspect_ratio = video_info.aspect_ratio
        
        # If already 4K or higher, maintain resolution
        if video_info.is_4k:
            return (width, height)
        
        # Calculate 4K target maintaining aspect ratio
        if aspect_ratio >= 16/9 - 0.1:  # 16:9 or wider
            return (3840, 2160)
        elif aspect_ratio >= 4/3 - 0.1:  # 4:3
            return (2880, 2160)
        else:  # Other ratios
            # Maintain aspect ratio, target 2160p height
            target_width = int(2160 * aspect_ratio)
            # Ensure width is even
            target_width = (target_width // 2) * 2
            return (target_width, 2160)