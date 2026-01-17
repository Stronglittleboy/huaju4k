"""
Conservative memory management implementation for huaju4k video enhancement.

This module provides intelligent memory management with adaptive tile sizing
and resource monitoring to prevent system crashes during video processing.
"""

import os
import logging
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from .interfaces import MemoryManager
from ..models.data_models import TileConfiguration, ResourceStatus
from ..utils.system_utils import get_memory_info, check_gpu_availability

logger = logging.getLogger(__name__)


@dataclass
class MemoryProfile:
    """Memory usage profile for different processing operations."""
    
    base_memory_mb: int  # Base memory usage
    per_tile_memory_mb: int  # Memory per tile
    gpu_memory_mb: int  # GPU memory requirement
    temp_file_overhead: float  # Temporary file overhead multiplier


class ConservativeMemoryManager(MemoryManager):
    """
    Conservative memory management with adaptive tile sizing.
    
    This implementation prioritizes system stability by using conservative
    memory estimates and adaptive tile sizing to prevent out-of-memory errors.
    """
    
    def __init__(self, safety_margin: float = 0.7, temp_dir: Optional[str] = None):
        """
        Initialize conservative memory manager.
        
        Args:
            safety_margin: Fraction of available memory to use (0.1-0.9)
            temp_dir: Directory for temporary files
        """
        if not 0.1 <= safety_margin <= 0.9:
            raise ValueError("Safety margin must be between 0.1 and 0.9")
        
        self.safety_margin = safety_margin
        self.temp_dir = Path(temp_dir) if temp_dir else Path.cwd() / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory profiles for different operations
        self.memory_profiles = {
            'ai_upscaling': MemoryProfile(
                base_memory_mb=500,
                per_tile_memory_mb=50,
                gpu_memory_mb=1000,
                temp_file_overhead=2.0
            ),
            'video_processing': MemoryProfile(
                base_memory_mb=200,
                per_tile_memory_mb=20,
                gpu_memory_mb=500,
                temp_file_overhead=1.5
            ),
            'audio_processing': MemoryProfile(
                base_memory_mb=100,
                per_tile_memory_mb=5,
                gpu_memory_mb=0,
                temp_file_overhead=1.2
            )
        }
        
        # Track allocated resources
        self._allocated_memory_mb = 0
        self._temp_files = []
        
        logger.info(f"ConservativeMemoryManager initialized with {safety_margin:.1%} safety margin")
    
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
        try:
            width, height = video_resolution
            
            # Get current memory status
            memory_info = get_memory_info()
            gpu_info = check_gpu_availability()
            
            available_memory_mb = memory_info.get('available_mb', 4000)  # Fallback to 4GB
            available_gpu_mb = gpu_info.get('gpu_memory_mb', 0)
            
            # Apply safety margin
            safe_memory_mb = int(available_memory_mb * self.safety_margin)
            safe_gpu_mb = int(available_gpu_mb * self.safety_margin) if available_gpu_mb > 0 else 0
            
            # Determine processing mode
            use_gpu = safe_gpu_mb >= model_memory_req and gpu_info.get('gpu_available', False)
            processing_mode = "gpu" if use_gpu else "cpu"
            
            # Select memory profile based on model requirements
            if model_memory_req > 800:
                profile = self.memory_profiles['ai_upscaling']
            elif model_memory_req > 200:
                profile = self.memory_profiles['video_processing']
            else:
                profile = self.memory_profiles['audio_processing']
            
            # Calculate available memory for tiles
            if use_gpu:
                available_for_tiles = safe_gpu_mb - profile.gpu_memory_mb
                memory_per_tile = profile.per_tile_memory_mb
            else:
                available_for_tiles = safe_memory_mb - profile.base_memory_mb
                memory_per_tile = profile.per_tile_memory_mb * 2  # CPU uses more memory
            
            # Ensure minimum available memory
            available_for_tiles = max(available_for_tiles, memory_per_tile)
            
            # Calculate maximum tiles that fit in memory
            max_tiles = max(1, available_for_tiles // memory_per_tile)
            
            # Calculate tile dimensions
            tile_width, tile_height = self._calculate_tile_dimensions(
                width, height, max_tiles
            )
            
            # Calculate batch size (how many tiles to process simultaneously)
            batch_size = min(max_tiles, self._calculate_optimal_batch_size(
                tile_width, tile_height, memory_per_tile, available_for_tiles
            ))
            
            # Calculate overlap for seamless processing
            overlap = self._calculate_overlap(tile_width, tile_height)
            
            # Estimate total memory usage
            estimated_memory = (
                profile.base_memory_mb + 
                (batch_size * memory_per_tile) +
                int(self._estimate_temp_file_memory(width, height) * profile.temp_file_overhead)
            )
            
            config = TileConfiguration(
                tile_width=tile_width,
                tile_height=tile_height,
                overlap=overlap,
                batch_size=batch_size,
                memory_usage_mb=estimated_memory,
                processing_mode=processing_mode
            )
            
            logger.info(f"Calculated tile configuration: {tile_width}x{tile_height}, "
                       f"batch_size={batch_size}, mode={processing_mode}, "
                       f"estimated_memory={estimated_memory}MB")
            
            return config
            
        except Exception as e:
            logger.error(f"Error calculating optimal tile size: {e}")
            # Return conservative fallback configuration
            return TileConfiguration(
                tile_width=512,
                tile_height=512,
                overlap=32,
                batch_size=1,
                memory_usage_mb=1000,
                processing_mode="cpu"
            )
    
    def monitor_and_adjust(self) -> ResourceStatus:
        """
        Monitor resource usage and adjust processing parameters.
        
        Returns:
            ResourceStatus containing current resource state
        """
        try:
            # Get current system status
            memory_info = get_memory_info()
            gpu_info = check_gpu_availability()
            
            # Calculate current usage
            available_memory = memory_info.get('available_mb', 0)
            total_memory = memory_info.get('total_mb', 0)
            memory_usage_percent = memory_info.get('memory_percent', 0)
            
            # Get GPU status if available
            gpu_memory_mb = 0
            gpu_usage_percent = 0.0
            if gpu_info.get('gpu_available', False):
                gpu_memory_mb = gpu_info.get('gpu_memory_mb', 0)
                # GPU usage would need additional monitoring (not implemented in basic version)
                gpu_usage_percent = 0.0
            
            # Check disk space
            disk_space_mb = self._get_available_disk_space()
            
            # Create resource status
            status = ResourceStatus(
                available_memory_mb=available_memory,
                available_gpu_memory_mb=gpu_memory_mb,
                cpu_usage_percent=100.0 - memory_usage_percent,  # Rough estimate
                gpu_usage_percent=gpu_usage_percent,
                disk_space_mb=disk_space_mb
            )
            
            # Log warnings if resources are low
            if status.memory_pressure == "high":
                logger.warning(f"High memory pressure: {available_memory}MB available")
            
            if disk_space_mb < 1000:  # Less than 1GB
                logger.warning(f"Low disk space: {disk_space_mb}MB available")
            
            return status
            
        except Exception as e:
            logger.error(f"Error monitoring resources: {e}")
            # Return safe fallback status
            return ResourceStatus(
                available_memory_mb=1000,
                available_gpu_memory_mb=0,
                cpu_usage_percent=50.0,
                gpu_usage_percent=0.0,
                disk_space_mb=5000
            )
    
    def cleanup_resources(self) -> None:
        """Release all allocated resources and clean temporary files."""
        try:
            # Clean up temporary files
            cleaned_files = 0
            for temp_file in self._temp_files[:]:  # Copy list to avoid modification during iteration
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                        cleaned_files += 1
                    self._temp_files.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_file}: {e}")
            
            # Clean up temp directory if empty
            try:
                if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
                    self.temp_dir.rmdir()
            except Exception as e:
                logger.debug(f"Could not remove temp directory: {e}")
            
            # Reset allocated memory tracking
            self._allocated_memory_mb = 0
            
            logger.info(f"Cleaned up {cleaned_files} temporary files")
            
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")
    
    def get_available_memory(self) -> Dict[str, int]:
        """
        Get available system and GPU memory.
        
        Returns:
            Dictionary with 'system' and 'gpu' memory in MB
        """
        try:
            memory_info = get_memory_info()
            gpu_info = check_gpu_availability()
            
            return {
                'system': memory_info.get('available_mb', 0),
                'gpu': gpu_info.get('gpu_memory_mb', 0) if gpu_info.get('gpu_available', False) else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting available memory: {e}")
            return {'system': 0, 'gpu': 0}
    
    def allocate_memory(self, size_mb: int, memory_type: str = 'system') -> bool:
        """
        Allocate memory for processing operations.
        
        Args:
            size_mb: Amount of memory to allocate in MB
            memory_type: Type of memory ('system' or 'gpu')
            
        Returns:
            True if allocation successful
            
        Raises:
            ValueError: If memory_type is invalid
        """
        try:
            available = self.get_available_memory()
            
            if memory_type == 'system':
                available_mb = available['system']
                safe_allocation = int(available_mb * self.safety_margin)
            elif memory_type == 'gpu':
                available_mb = available['gpu']
                safe_allocation = int(available_mb * self.safety_margin)
            else:
                raise ValueError(f"Unknown memory type: {memory_type}")
            
            if size_mb <= safe_allocation:
                self._allocated_memory_mb += size_mb
                logger.debug(f"Allocated {size_mb}MB of {memory_type} memory")
                return True
            else:
                logger.warning(f"Cannot allocate {size_mb}MB of {memory_type} memory "
                             f"(available: {safe_allocation}MB)")
                return False
                
        except ValueError:
            # Re-raise ValueError for invalid memory types
            raise
        except Exception as e:
            logger.error(f"Error allocating memory: {e}")
            return False
    
    def create_temp_file(self, prefix: str = "huaju4k_", suffix: str = ".tmp") -> Path:
        """
        Create a temporary file and track it for cleanup.
        
        Args:
            prefix: File name prefix
            suffix: File name suffix
            
        Returns:
            Path to created temporary file
        """
        try:
            import tempfile
            
            # Create temporary file
            fd, temp_path = tempfile.mkstemp(
                prefix=prefix, 
                suffix=suffix, 
                dir=str(self.temp_dir)
            )
            os.close(fd)  # Close file descriptor
            
            temp_file = Path(temp_path)
            self._temp_files.append(temp_file)
            
            logger.debug(f"Created temporary file: {temp_file}")
            return temp_file
            
        except Exception as e:
            logger.error(f"Error creating temporary file: {e}")
            # Return fallback path
            fallback_path = self.temp_dir / f"{prefix}fallback{suffix}"
            self._temp_files.append(fallback_path)
            return fallback_path
    
    def _calculate_tile_dimensions(self, width: int, height: int, max_tiles: int) -> Tuple[int, int]:
        """Calculate optimal tile dimensions for given constraints."""
        # Start with square tiles
        aspect_ratio = width / height
        
        # Calculate tile size based on available tiles
        total_pixels = width * height
        pixels_per_tile = total_pixels / max_tiles
        
        # Calculate square tile size
        base_tile_size = int((pixels_per_tile) ** 0.5)
        
        # Adjust for aspect ratio
        if aspect_ratio > 1:  # Wider than tall
            tile_width = int(base_tile_size * (aspect_ratio ** 0.5))
            tile_height = int(base_tile_size / (aspect_ratio ** 0.5))
        else:  # Taller than wide
            tile_width = int(base_tile_size / ((1/aspect_ratio) ** 0.5))
            tile_height = int(base_tile_size * ((1/aspect_ratio) ** 0.5))
        
        # Ensure tiles are not larger than video dimensions
        tile_width = min(tile_width, width)
        tile_height = min(tile_height, height)
        
        # Ensure minimum tile size
        tile_width = max(tile_width, 64)
        tile_height = max(tile_height, 64)
        
        # Round to multiples of 8 for better processing
        tile_width = (tile_width // 8) * 8
        tile_height = (tile_height // 8) * 8
        
        return tile_width, tile_height
    
    def _calculate_optimal_batch_size(self, tile_width: int, tile_height: int,
                                    memory_per_tile: int, available_memory: int) -> int:
        """Calculate optimal batch size for tile processing."""
        # Calculate how many tiles can fit in available memory
        max_batch = max(1, available_memory // memory_per_tile)
        
        # Limit batch size based on tile size (larger tiles = smaller batches)
        tile_pixels = tile_width * tile_height
        if tile_pixels > 1024 * 1024:  # Large tiles (>1MP)
            max_batch = min(max_batch, 2)
        elif tile_pixels > 512 * 512:  # Medium tiles (>0.25MP)
            max_batch = min(max_batch, 4)
        else:  # Small tiles
            max_batch = min(max_batch, 8)
        
        return max_batch
    
    def _calculate_overlap(self, tile_width: int, tile_height: int) -> int:
        """Calculate overlap size for seamless tile processing."""
        # Use 5% of the smaller dimension, with min/max bounds
        smaller_dim = min(tile_width, tile_height)
        overlap = int(smaller_dim * 0.05)
        
        # Ensure overlap is within reasonable bounds
        overlap = max(16, min(overlap, 128))
        
        # Round to multiple of 8
        overlap = (overlap // 8) * 8
        
        return overlap
    
    def _estimate_temp_file_memory(self, width: int, height: int) -> int:
        """Estimate memory usage for temporary files."""
        # Estimate based on video resolution
        # Assume 3 bytes per pixel (RGB) for temporary frame storage
        pixels = width * height
        estimated_mb = (pixels * 3) // (1024 * 1024)
        
        # Add buffer for multiple frames and processing overhead
        return max(100, estimated_mb * 2)
    
    def _get_available_disk_space(self) -> int:
        """Get available disk space in MB."""
        try:
            import shutil
            free_space = shutil.disk_usage(self.temp_dir).free
            return int(free_space / (1024 * 1024))
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return 5000  # Fallback to 5GB