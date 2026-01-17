"""
Tile-based processing system for huaju4k video enhancement.

This module provides adaptive tile processing with memory-safe batching
and overlap handling for seamless AI upscaling results.
"""

import logging
from typing import List, Tuple, Optional, Iterator, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

import numpy as np

from ..models.data_models import TileConfiguration, ProcessingStrategy
from .ai_model_manager import AIModelManager
from .memory_manager import ConservativeMemoryManager

logger = logging.getLogger(__name__)


@dataclass
class TileInfo:
    """Information about a processing tile."""
    
    x: int  # X coordinate in original image
    y: int  # Y coordinate in original image
    width: int  # Tile width
    height: int  # Tile height
    overlap_left: int  # Left overlap in pixels
    overlap_right: int  # Right overlap in pixels
    overlap_top: int  # Top overlap in pixels
    overlap_bottom: int  # Bottom overlap in pixels
    tile_id: int  # Unique tile identifier
    
    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get tile bounds as (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    @property
    def effective_bounds(self) -> Tuple[int, int, int, int]:
        """Get effective bounds without overlap."""
        x1 = self.x + self.overlap_left
        y1 = self.y + self.overlap_top
        x2 = self.x + self.width - self.overlap_right
        y2 = self.y + self.height - self.overlap_bottom
        return (x1, y1, x2, y2)


@dataclass
class ProcessedTile:
    """Result of tile processing."""
    
    tile_info: TileInfo
    processed_data: np.ndarray
    processing_time: float
    success: bool
    error: Optional[str] = None


class TileGenerator:
    """
    Generator for creating processing tiles with adaptive sizing and overlap.
    
    This class handles the complex logic of dividing images into overlapping
    tiles that can be processed independently and then seamlessly reassembled.
    """
    
    def __init__(self, tile_config: TileConfiguration):
        """
        Initialize tile generator.
        
        Args:
            tile_config: Configuration for tile processing
        """
        self.config = tile_config
        self.logger = logging.getLogger(f"{__name__}.TileGenerator")
    
    def generate_tiles(self, image_shape: Tuple[int, int]) -> List[TileInfo]:
        """
        Generate tiles for an image with given shape.
        
        Args:
            image_shape: Image shape as (height, width)
            
        Returns:
            List of TileInfo objects
        """
        height, width = image_shape
        tile_width = self.config.tile_width
        tile_height = self.config.tile_height
        overlap = self.config.overlap
        
        tiles = []
        tile_id = 0
        
        # Calculate number of tiles needed
        tiles_x = self._calculate_tiles_needed(width, tile_width, overlap)
        tiles_y = self._calculate_tiles_needed(height, tile_height, overlap)
        
        self.logger.info(f"Generating {tiles_x}x{tiles_y} = {tiles_x * tiles_y} tiles "
                        f"for image {width}x{height}")
        
        for row in range(tiles_y):
            for col in range(tiles_x):
                tile_info = self._create_tile_info(
                    col, row, tiles_x, tiles_y,
                    width, height, tile_width, tile_height, overlap, tile_id
                )
                tiles.append(tile_info)
                tile_id += 1
        
        return tiles
    
    def _calculate_tiles_needed(self, dimension: int, tile_size: int, overlap: int) -> int:
        """Calculate number of tiles needed for a dimension."""
        if dimension <= tile_size:
            return 1
        
        effective_tile_size = tile_size - overlap
        remaining = dimension - tile_size
        additional_tiles = (remaining + effective_tile_size - 1) // effective_tile_size
        
        return 1 + additional_tiles
    
    def _create_tile_info(self, col: int, row: int, total_cols: int, total_rows: int,
                         image_width: int, image_height: int,
                         tile_width: int, tile_height: int, overlap: int, tile_id: int) -> TileInfo:
        """Create TileInfo for specific position."""
        
        # Calculate base position
        effective_tile_width = tile_width - overlap
        effective_tile_height = tile_height - overlap
        
        x = col * effective_tile_width
        y = row * effective_tile_height
        
        # Adjust for image boundaries
        actual_width = min(tile_width, image_width - x)
        actual_height = min(tile_height, image_height - y)
        
        # Calculate overlaps
        overlap_left = overlap // 2 if col > 0 else 0
        overlap_right = overlap // 2 if col < total_cols - 1 else 0
        overlap_top = overlap // 2 if row > 0 else 0
        overlap_bottom = overlap // 2 if row < total_rows - 1 else 0
        
        # Adjust position for overlap
        x = max(0, x - overlap_left)
        y = max(0, y - overlap_top)
        
        # Adjust size for overlap
        actual_width = min(actual_width + overlap_left + overlap_right, image_width - x)
        actual_height = min(actual_height + overlap_top + overlap_bottom, image_height - y)
        
        return TileInfo(
            x=x, y=y,
            width=actual_width, height=actual_height,
            overlap_left=overlap_left, overlap_right=overlap_right,
            overlap_top=overlap_top, overlap_bottom=overlap_bottom,
            tile_id=tile_id
        )


class TileAssembler:
    """
    Assembler for combining processed tiles back into complete images.
    
    This class handles the complex logic of blending overlapping tiles
    to create seamless results without visible tile boundaries.
    """
    
    def __init__(self, scale_factor: int = 4):
        """
        Initialize tile assembler.
        
        Args:
            scale_factor: Upscaling factor for output size calculation
        """
        self.scale_factor = scale_factor
        self.logger = logging.getLogger(f"{__name__}.TileAssembler")
    
    def assemble_tiles(self, processed_tiles: List[ProcessedTile], 
                      original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Assemble processed tiles into complete image.
        
        Args:
            processed_tiles: List of processed tiles
            original_shape: Original image shape (height, width)
            
        Returns:
            Assembled image as numpy array
        """
        original_height, original_width = original_shape
        output_height = original_height * self.scale_factor
        output_width = original_width * self.scale_factor
        
        # Determine number of channels from first successful tile
        channels = 3  # Default to RGB
        for tile in processed_tiles:
            if tile.success and tile.processed_data is not None:
                channels = tile.processed_data.shape[2] if len(tile.processed_data.shape) == 3 else 1
                break
        
        # Initialize output image
        if channels == 1:
            output_image = np.zeros((output_height, output_width), dtype=np.uint8)
        else:
            output_image = np.zeros((output_height, output_width, channels), dtype=np.uint8)
        
        # Initialize weight map for blending
        weight_map = np.zeros((output_height, output_width), dtype=np.float32)
        
        successful_tiles = 0
        failed_tiles = 0
        
        for tile in processed_tiles:
            if not tile.success or tile.processed_data is None:
                failed_tiles += 1
                self.logger.warning(f"Skipping failed tile {tile.tile_info.tile_id}: {tile.error}")
                continue
            
            self._blend_tile(output_image, weight_map, tile)
            successful_tiles += 1
        
        # Normalize by weights
        self._normalize_output(output_image, weight_map)
        
        self.logger.info(f"Assembled {successful_tiles} tiles successfully, {failed_tiles} failed")
        
        return output_image
    
    def _blend_tile(self, output_image: np.ndarray, weight_map: np.ndarray, 
                   tile: ProcessedTile) -> None:
        """Blend a single tile into the output image."""
        tile_info = tile.tile_info
        tile_data = tile.processed_data
        
        # Calculate output coordinates (scaled)
        out_x1 = tile_info.x * self.scale_factor
        out_y1 = tile_info.y * self.scale_factor
        out_x2 = out_x1 + tile_data.shape[1]
        out_y2 = out_y1 + tile_data.shape[0]
        
        # Ensure coordinates are within bounds
        out_x1 = max(0, out_x1)
        out_y1 = max(0, out_y1)
        out_x2 = min(output_image.shape[1], out_x2)
        out_y2 = min(output_image.shape[0], out_y2)
        
        # Calculate tile coordinates
        tile_x1 = max(0, -tile_info.x * self.scale_factor) if tile_info.x < 0 else 0
        tile_y1 = max(0, -tile_info.y * self.scale_factor) if tile_info.y < 0 else 0
        tile_x2 = tile_x1 + (out_x2 - out_x1)
        tile_y2 = tile_y1 + (out_y2 - out_y1)
        
        # Create weight mask for smooth blending
        weight_mask = self._create_weight_mask(
            tile_data.shape[:2], tile_info, self.scale_factor
        )
        
        # Extract regions
        output_region = output_image[out_y1:out_y2, out_x1:out_x2]
        weight_region = weight_map[out_y1:out_y2, out_x1:out_x2]
        tile_region = tile_data[tile_y1:tile_y2, tile_x1:tile_x2]
        weight_region_tile = weight_mask[tile_y1:tile_y2, tile_x1:tile_x2]
        
        # Ensure shapes match
        if output_region.shape[:2] != tile_region.shape[:2]:
            self.logger.warning(f"Shape mismatch for tile {tile_info.tile_id}: "
                              f"output {output_region.shape} vs tile {tile_region.shape}")
            return
        
        # Blend using weighted average
        if len(output_image.shape) == 3:  # Color image
            for c in range(output_image.shape[2]):
                output_region[:, :, c] = (
                    output_region[:, :, c] * weight_region +
                    tile_region[:, :, c] * weight_region_tile
                ) / np.maximum(weight_region + weight_region_tile, 1e-8)
        else:  # Grayscale image
            output_region[:, :] = (
                output_region * weight_region +
                tile_region * weight_region_tile
            ) / np.maximum(weight_region + weight_region_tile, 1e-8)
        
        # Update weight map
        weight_region += weight_region_tile
    
    def _create_weight_mask(self, tile_shape: Tuple[int, int], 
                           tile_info: TileInfo, scale_factor: int) -> np.ndarray:
        """Create weight mask for smooth tile blending."""
        height, width = tile_shape
        mask = np.ones((height, width), dtype=np.float32)
        
        # Calculate overlap regions in scaled coordinates
        overlap_left = tile_info.overlap_left * scale_factor
        overlap_right = tile_info.overlap_right * scale_factor
        overlap_top = tile_info.overlap_top * scale_factor
        overlap_bottom = tile_info.overlap_bottom * scale_factor
        
        # Apply feathering to overlap regions
        if overlap_left > 0:
            for i in range(min(overlap_left, width)):
                weight = i / overlap_left
                mask[:, i] *= weight
        
        if overlap_right > 0:
            for i in range(min(overlap_right, width)):
                col = width - 1 - i
                weight = i / overlap_right
                mask[:, col] *= weight
        
        if overlap_top > 0:
            for i in range(min(overlap_top, height)):
                weight = i / overlap_top
                mask[i, :] *= weight
        
        if overlap_bottom > 0:
            for i in range(min(overlap_bottom, height)):
                row = height - 1 - i
                weight = i / overlap_bottom
                mask[row, :] *= weight
        
        return mask
    
    def _normalize_output(self, output_image: np.ndarray, weight_map: np.ndarray) -> None:
        """Normalize output image by weight map."""
        # Avoid division by zero
        weight_map = np.maximum(weight_map, 1e-8)
        
        if len(output_image.shape) == 3:  # Color image
            for c in range(output_image.shape[2]):
                output_image[:, :, c] = output_image[:, :, c] / weight_map
        else:  # Grayscale image
            output_image[:, :] = output_image / weight_map


class TileProcessor:
    """
    Main tile-based processing system with memory-safe batching.
    
    This class orchestrates the entire tile processing pipeline including
    tile generation, batch processing, and result assembly.
    """
    
    def __init__(self, memory_manager: ConservativeMemoryManager, 
                 progress_tracker=None, ai_model_manager: AIModelManager = None):
        """
        Initialize tile processor.
        
        Args:
            memory_manager: Memory manager instance
            progress_tracker: Progress tracker instance (optional)
            ai_model_manager: AI model manager instance (optional, will be set later)
        """
        self.memory_manager = memory_manager
        self.progress_tracker = progress_tracker
        self.ai_model_manager = ai_model_manager
        self.logger = logging.getLogger(f"{__name__}.TileProcessor")
        
        # Processing statistics
        self.stats = {
            'total_tiles': 0,
            'successful_tiles': 0,
            'failed_tiles': 0,
            'total_processing_time': 0.0,
            'average_tile_time': 0.0
        }
    
    def set_ai_model_manager(self, ai_model_manager: AIModelManager) -> None:
        """Set AI model manager after initialization."""
        self.ai_model_manager = ai_model_manager
    
    def process_image_tiles(self, image: np.ndarray, 
                           tile_config: TileConfiguration,
                           progress_callback: Optional[callable] = None) -> np.ndarray:
        """
        Process image using tile-based approach.
        
        Args:
            image: Input image as numpy array (H, W, C)
            tile_config: Tile processing configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processed image as numpy array
        """
        start_time = time.time()
        
        try:
            # Check if AI model manager is available
            if not self.ai_model_manager:
                self.logger.error("AI model manager not set")
                return image
            
            # Generate tiles
            tile_generator = TileGenerator(tile_config)
            tiles = tile_generator.generate_tiles(image.shape[:2])
            
            self.stats['total_tiles'] = len(tiles)
            self.logger.info(f"Processing {len(tiles)} tiles with batch size {tile_config.batch_size}")
            
            # Process tiles in batches
            processed_tiles = self._process_tiles_in_batches(
                image, tiles, tile_config, progress_callback
            )
            
            # Assemble results
            scale_factor = 4  # TODO: Get from model info
            assembler = TileAssembler(scale_factor)
            result = assembler.assemble_tiles(processed_tiles, image.shape[:2])
            
            # Update statistics
            total_time = time.time() - start_time
            self.stats['total_processing_time'] = total_time
            if self.stats['successful_tiles'] > 0:
                self.stats['average_tile_time'] = total_time / self.stats['successful_tiles']
            
            self.logger.info(f"Tile processing completed in {total_time:.2f}s, "
                           f"success rate: {self.stats['successful_tiles']}/{self.stats['total_tiles']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in tile processing: {e}")
            # Return original image as fallback
            return image
    
    def _process_tiles_in_batches(self, image: np.ndarray, tiles: List[TileInfo],
                                 tile_config: TileConfiguration,
                                 progress_callback: Optional[callable] = None) -> List[ProcessedTile]:
        """Process tiles in memory-safe batches."""
        processed_tiles = []
        batch_size = tile_config.batch_size
        
        # Process tiles in batches
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i + batch_size]
            
            # Check memory before processing batch
            resource_status = self.memory_manager.monitor_and_adjust()
            if resource_status.memory_pressure == "high":
                self.logger.warning("High memory pressure, reducing batch size")
                batch_tiles = batch_tiles[:max(1, len(batch_tiles) // 2)]
            
            # Extract tile images
            tile_images = []
            for tile_info in batch_tiles:
                tile_image = self._extract_tile_image(image, tile_info)
                tile_images.append((tile_info, tile_image))
            
            # Process batch
            batch_results = self._process_tile_batch(tile_images)
            processed_tiles.extend(batch_results)
            
            # Update progress
            if progress_callback:
                progress = (i + len(batch_tiles)) / len(tiles)
                progress_callback(progress, f"Processed {i + len(batch_tiles)}/{len(tiles)} tiles")
            
            # Log batch completion
            successful_in_batch = sum(1 for result in batch_results if result.success)
            self.logger.debug(f"Batch {i//batch_size + 1}: {successful_in_batch}/{len(batch_tiles)} successful")
        
        return processed_tiles
    
    def _extract_tile_image(self, image: np.ndarray, tile_info: TileInfo) -> np.ndarray:
        """Extract tile image from full image."""
        x1, y1, x2, y2 = tile_info.bounds
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        tile_image = image[y1:y2, x1:x2].copy()
        
        return tile_image
    
    def _process_tile_batch(self, tile_batch: List[Tuple[TileInfo, np.ndarray]]) -> List[ProcessedTile]:
        """Process a batch of tiles."""
        results = []
        
        # Extract images for batch processing
        images = [tile_image for _, tile_image in tile_batch]
        
        try:
            # Process batch with AI model
            start_time = time.time()
            processed_images = self.ai_model_manager.predict_batch(images)
            processing_time = time.time() - start_time
            
            # Create results
            for i, (tile_info, _) in enumerate(tile_batch):
                if i < len(processed_images):
                    result = ProcessedTile(
                        tile_info=tile_info,
                        processed_data=processed_images[i],
                        processing_time=processing_time / len(tile_batch),
                        success=True
                    )
                    self.stats['successful_tiles'] += 1
                else:
                    result = ProcessedTile(
                        tile_info=tile_info,
                        processed_data=None,
                        processing_time=0.0,
                        success=False,
                        error="Missing result from batch processing"
                    )
                    self.stats['failed_tiles'] += 1
                
                results.append(result)
                
        except Exception as e:
            # Handle batch processing failure
            self.logger.error(f"Batch processing failed: {e}")
            
            for tile_info, tile_image in tile_batch:
                result = ProcessedTile(
                    tile_info=tile_info,
                    processed_data=tile_image,  # Return original as fallback
                    processing_time=0.0,
                    success=False,
                    error=str(e)
                )
                results.append(result)
                self.stats['failed_tiles'] += 1
        
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'total_tiles': 0,
            'successful_tiles': 0,
            'failed_tiles': 0,
            'total_processing_time': 0.0,
            'average_tile_time': 0.0
        }