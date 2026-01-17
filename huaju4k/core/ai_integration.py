"""
AI integration system for huaju4k video enhancement.

This module provides a high-level interface that integrates AI model management
and tile-based processing for seamless video enhancement operations.
"""

import logging
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

import numpy as np

from .ai_model_manager import AIModelManager
from .tile_processor import TileProcessor
from .memory_manager import ConservativeMemoryManager
from ..models.data_models import ProcessingStrategy, TileConfiguration, ProcessResult

logger = logging.getLogger(__name__)


class AIVideoProcessor:
    """
    High-level AI video processing system.
    
    This class provides a unified interface for AI-based video enhancement
    that handles model management, tile processing, and resource optimization.
    """
    
    def __init__(self, models_dir: str = "./models", temp_dir: Optional[str] = None):
        """
        Initialize AI video processor.
        
        Args:
            models_dir: Directory containing AI model files
            temp_dir: Directory for temporary files
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.memory_manager = ConservativeMemoryManager(temp_dir=temp_dir)
        self.ai_model_manager = AIModelManager(str(self.models_dir))
        self.tile_processor = TileProcessor(self.ai_model_manager, self.memory_manager)
        
        # Current processing state
        self.current_model = None
        self.current_strategy = None
        
        logger.info("AIVideoProcessor initialized")
    
    def initialize_processing(self, strategy: ProcessingStrategy) -> bool:
        """
        Initialize processing with given strategy.
        
        Args:
            strategy: Processing strategy configuration
            
        Returns:
            True if initialization successful
        """
        try:
            # Load AI model
            model_loaded = self.ai_model_manager.load_model(
                strategy.ai_model, 
                strategy.use_gpu
            )
            
            if not model_loaded:
                # Try to auto-select a suitable model
                available_memory = self.memory_manager.get_available_memory()
                system_memory = available_memory.get('system', 0)
                
                auto_model = self.ai_model_manager.auto_select_model(
                    strategy.target_resolution, 
                    system_memory
                )
                
                if auto_model:
                    logger.info(f"Auto-selected model: {auto_model}")
                    model_loaded = self.ai_model_manager.load_model(auto_model, strategy.use_gpu)
                    # Update strategy with selected model
                    strategy.ai_model = auto_model
                
                if not model_loaded:
                    logger.error("Failed to load any AI model")
                    return False
            
            self.current_model = strategy.ai_model
            self.current_strategy = strategy
            
            logger.info(f"Processing initialized with model: {self.current_model}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing processing: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, 
                     progress_callback: Optional[callable] = None) -> np.ndarray:
        """
        Process a single video frame.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Enhanced frame as numpy array
        """
        if self.current_strategy is None:
            raise RuntimeError("Processing not initialized. Call initialize_processing() first.")
        
        try:
            # Get tile configuration
            tile_config = self._get_tile_configuration(frame.shape[:2])
            
            # Process frame using tiles
            enhanced_frame = self.tile_processor.process_image_tiles(
                frame, tile_config, progress_callback
            )
            
            return enhanced_frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            # Return original frame as fallback
            return frame
    
    def process_frame_batch(self, frames: List[np.ndarray],
                           progress_callback: Optional[callable] = None) -> List[np.ndarray]:
        """
        Process a batch of video frames.
        
        Args:
            frames: List of input frames
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of enhanced frames
        """
        if self.current_strategy is None:
            raise RuntimeError("Processing not initialized. Call initialize_processing() first.")
        
        enhanced_frames = []
        
        for i, frame in enumerate(frames):
            try:
                enhanced_frame = self.process_frame(frame)
                enhanced_frames.append(enhanced_frame)
                
                # Update progress
                if progress_callback:
                    progress = (i + 1) / len(frames)
                    progress_callback(progress, f"Processed frame {i + 1}/{len(frames)}")
                    
            except Exception as e:
                logger.error(f"Error processing frame {i}: {e}")
                # Use original frame as fallback
                enhanced_frames.append(frame)
        
        return enhanced_frames
    
    def get_processing_info(self) -> Dict[str, Any]:
        """
        Get information about current processing configuration.
        
        Returns:
            Dictionary containing processing information
        """
        info = {
            'current_model': self.current_model,
            'available_models': self.ai_model_manager.get_available_models(),
            'memory_usage': self.ai_model_manager.get_memory_usage(),
            'resource_status': self.memory_manager.monitor_and_adjust(),
            'processing_stats': self.tile_processor.get_processing_stats()
        }
        
        if self.current_strategy:
            info['current_strategy'] = {
                'ai_model': self.current_strategy.ai_model,
                'use_gpu': self.current_strategy.use_gpu,
                'quality_preset': self.current_strategy.quality_preset,
                'target_resolution': self.current_strategy.target_resolution
            }
        
        return info
    
    def optimize_for_video(self, video_resolution: Tuple[int, int], 
                          duration: float, framerate: float) -> ProcessingStrategy:
        """
        Create optimized processing strategy for video characteristics.
        
        Args:
            video_resolution: Video resolution (width, height)
            duration: Video duration in seconds
            framerate: Video framerate
            
        Returns:
            Optimized ProcessingStrategy
        """
        try:
            # Get available memory
            available_memory = self.memory_manager.get_available_memory()
            system_memory = available_memory.get('system', 4000)
            gpu_memory = available_memory.get('gpu', 0)
            
            # Auto-select model based on constraints
            auto_model = self.ai_model_manager.auto_select_model(
                video_resolution, system_memory
            )
            
            if auto_model is None:
                auto_model = 'opencv_cubic'  # Safe fallback
            
            # Determine if GPU should be used
            use_gpu = gpu_memory > 1000 and 'real_esrgan' in auto_model
            
            # Calculate target resolution (4K)
            target_width = 3840
            target_height = int(target_width * video_resolution[1] / video_resolution[0])
            
            # Determine quality preset based on video characteristics
            total_frames = duration * framerate
            if total_frames > 10000:  # Long video
                quality_preset = 'fast'
            elif total_frames > 3000:  # Medium video
                quality_preset = 'balanced'
            else:  # Short video
                quality_preset = 'high'
            
            # Get tile configuration
            model_info = self.ai_model_manager.get_model_info(auto_model)
            memory_req = model_info.memory_requirement_mb if model_info else 500
            
            tile_config = self.memory_manager.calculate_optimal_tile_size(
                video_resolution, memory_req
            )
            
            # Create processing strategy
            strategy = ProcessingStrategy(
                tile_size=(tile_config.tile_width, tile_config.tile_height),
                overlap_pixels=tile_config.overlap,
                batch_size=tile_config.batch_size,
                use_gpu=use_gpu,
                ai_model=auto_model,
                quality_preset=quality_preset,
                memory_limit_mb=int(system_memory * 0.7),
                target_resolution=(target_width, target_height),
                denoise_strength=0.7
            )
            
            logger.info(f"Optimized strategy: model={auto_model}, gpu={use_gpu}, "
                       f"quality={quality_preset}, tiles={tile_config.tile_width}x{tile_config.tile_height}")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error optimizing strategy: {e}")
            # Return safe fallback strategy
            return ProcessingStrategy(
                tile_size=(512, 512),
                overlap_pixels=32,
                batch_size=1,
                use_gpu=False,
                ai_model='opencv_cubic',
                quality_preset='balanced',
                memory_limit_mb=2000,
                target_resolution=(3840, 2160),
                denoise_strength=0.5
            )
    
    def cleanup(self) -> None:
        """Clean up resources and temporary files."""
        try:
            # Unload models
            self.ai_model_manager.clear_cache()
            
            # Clean up memory manager
            self.memory_manager.cleanup_resources()
            
            # Reset state
            self.current_model = None
            self.current_strategy = None
            
            logger.info("AIVideoProcessor cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _get_tile_configuration(self, frame_shape: Tuple[int, int]) -> TileConfiguration:
        """Get tile configuration for current strategy."""
        if self.current_strategy is None:
            raise RuntimeError("No current strategy available")
        
        return TileConfiguration(
            tile_width=self.current_strategy.tile_size[0],
            tile_height=self.current_strategy.tile_size[1],
            overlap=self.current_strategy.overlap_pixels,
            batch_size=self.current_strategy.batch_size,
            memory_usage_mb=self.current_strategy.memory_limit_mb,
            processing_mode="gpu" if self.current_strategy.use_gpu else "cpu"
        )
    
    def validate_model_availability(self) -> Dict[str, bool]:
        """
        Validate availability of AI models.
        
        Returns:
            Dictionary mapping model names to availability status
        """
        availability = {}
        
        for model_name in self.ai_model_manager.get_available_models():
            try:
                # Try to get model info
                model_info = self.ai_model_manager.get_model_info(model_name)
                if model_info:
                    if model_info.path == 'builtin':
                        availability[model_name] = True
                    else:
                        availability[model_name] = Path(model_info.path).exists()
                else:
                    availability[model_name] = False
            except Exception:
                availability[model_name] = False
        
        return availability