"""
AI model management system for huaju4k video enhancement.

This module provides model loading, caching, and GPU/CPU fallback mechanisms
for AI-based video upscaling operations.
"""

import os
import logging
import hashlib
import time
import gc
from typing import Dict, Optional, Any, Tuple, List
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import cv2

from ..models.data_models import (
    ProcessingStrategy, TileConfiguration, EnhancementStrategy,
    StructureFeatures, GANPolicy, TemporalConfig, MemoryConfig
)
from ..utils.system_utils import check_gpu_availability

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about an AI model."""
    
    name: str
    path: str
    scale_factor: int  # Upscaling factor (2x, 4x, etc.)
    input_channels: int  # Number of input channels (3 for RGB)
    tile_size: int  # Recommended tile size
    memory_requirement_mb: int  # Estimated memory requirement
    supports_gpu: bool
    model_hash: Optional[str] = None
    
    def __post_init__(self):
        """Calculate model hash if not provided."""
        if self.model_hash is None and Path(self.path).exists():
            self.model_hash = self._calculate_file_hash(self.path)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of model file."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate hash for {file_path}: {e}")
            return "unknown"


class AIModel(ABC):
    """Abstract base class for AI upscaling models."""
    
    @abstractmethod
    def load(self, model_path: str, use_gpu: bool = True) -> bool:
        """
        Load the AI model.
        
        Args:
            model_path: Path to model file
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            True if model loaded successfully
        """
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Perform upscaling prediction on image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Upscaled image as numpy array
        """
        pass
    
    @abstractmethod
    def predict_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Perform batch prediction on multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of upscaled images
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free resources."""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> int:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        pass


class RealESRGANModel(AIModel):
    """Real-ESRGAN model implementation."""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.is_loaded = False
        self.use_gpu = False
        
    def load(self, model_path: str, use_gpu: bool = True) -> bool:
        """Load Real-ESRGAN model."""
        try:
            # Check if Real-ESRGAN is available
            try:
                from realesrgan import RealESRGANer
                from basicsr.archs.rrdbnet_arch import RRDBNet
            except ImportError:
                logger.error("Real-ESRGAN not available. Install with: pip install realesrgan")
                return False
            
            # Check GPU availability
            gpu_available = check_gpu_availability().get('gpu_available', False)
            self.use_gpu = use_gpu and gpu_available
            
            # Determine device
            if self.use_gpu:
                try:
                    import torch
                    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                except ImportError:
                    logger.warning("PyTorch not available, falling back to CPU")
                    self.device = 'cpu'
                    self.use_gpu = False
            else:
                self.device = 'cpu'
            
            # Load model
            if not Path(model_path).exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Create RRDBNet model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )
            
            # Initialize upsampler
            self.model = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=512,  # Default tile size
                tile_pad=10,
                pre_pad=0,
                half=self.use_gpu,  # Use half precision on GPU
                device=self.device
            )
            
            self.is_loaded = True
            logger.info(f"Real-ESRGAN model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Real-ESRGAN model: {e}")
            self.is_loaded = False
            return False
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Perform upscaling prediction."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Real-ESRGAN expects BGR format
            if image.shape[2] == 3:  # RGB to BGR
                image_bgr = image[:, :, ::-1]
            else:
                image_bgr = image
            
            # Perform enhancement
            output, _ = self.model.enhance(image_bgr, outscale=4)
            
            # Convert back to RGB
            if output.shape[2] == 3:  # BGR to RGB
                output = output[:, :, ::-1]
            
            return output
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return input image as fallback
            return image
    
    def predict_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Perform batch prediction."""
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results
    
    def unload(self) -> None:
        """Unload model and free resources."""
        if self.model is not None:
            try:
                # Clear GPU memory if using GPU
                if self.use_gpu:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                self.model = None
                self.is_loaded = False
                logger.info("Real-ESRGAN model unloaded")
                
            except Exception as e:
                logger.warning(f"Error during model unload: {e}")
    
    def get_memory_usage(self) -> int:
        """Get estimated memory usage."""
        if not self.is_loaded:
            return 0
        
        # Rough estimate based on model size and device
        base_memory = 1000  # Base model memory in MB
        if self.use_gpu:
            return base_memory
        else:
            return base_memory // 2  # CPU uses less memory


class OpenCVModel(AIModel):
    """OpenCV-based upscaling model (fallback)."""
    
    def __init__(self):
        self.is_loaded = False
        self.scale_factor = 4
    
    def load(self, model_path: str, use_gpu: bool = True) -> bool:
        """Load OpenCV model (no actual loading needed)."""
        try:
            import cv2
            self.is_loaded = True
            logger.info("OpenCV fallback model loaded")
            return True
        except ImportError:
            logger.error("OpenCV not available")
            return False
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Perform upscaling using OpenCV."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            import cv2
            
            height, width = image.shape[:2]
            new_height = height * self.scale_factor
            new_width = width * self.scale_factor
            
            # Use INTER_CUBIC for better quality
            upscaled = cv2.resize(
                image, 
                (new_width, new_height), 
                interpolation=cv2.INTER_CUBIC
            )
            
            return upscaled
            
        except Exception as e:
            logger.error(f"OpenCV upscaling failed: {e}")
            return image
    
    def predict_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Perform batch prediction."""
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results
    
    def unload(self) -> None:
        """Unload model."""
        self.is_loaded = False
        logger.info("OpenCV model unloaded")
    
    def get_memory_usage(self) -> int:
        """Get memory usage (minimal for OpenCV)."""
        return 50 if self.is_loaded else 0


class AIModelManager:
    """
    AI model management system with caching and fallback mechanisms.
    
    This class handles loading, caching, and switching between different
    AI models for video upscaling operations.
    """
    
    def __init__(self, models_dir: str = "./models", cache_size: int = 2):
        """
        Initialize AI model manager.
        
        Args:
            models_dir: Directory containing model files
            cache_size: Maximum number of models to keep in cache
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_size = cache_size
        self.model_cache: Dict[str, AIModel] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        self.current_model: Optional[AIModel] = None
        self.current_model_name: Optional[str] = None
        
        # Register available models
        self._register_models()
        
        logger.info(f"AIModelManager initialized with cache size {cache_size}")
    
    def _register_models(self) -> None:
        """Register available AI models."""
        # Real-ESRGAN models
        realesrgan_models = [
            {
                'name': 'real_esrgan_x4',
                'filename': 'RealESRGAN_x4plus.pth',
                'scale_factor': 4,
                'tile_size': 512,
                'memory_mb': 1000
            },
            {
                'name': 'real_esrgan_x2',
                'filename': 'RealESRGAN_x2plus.pth',
                'scale_factor': 2,
                'tile_size': 768,
                'memory_mb': 800
            }
        ]
        
        for model_config in realesrgan_models:
            model_path = self.models_dir / model_config['filename']
            self.model_info[model_config['name']] = ModelInfo(
                name=model_config['name'],
                path=str(model_path),
                scale_factor=model_config['scale_factor'],
                input_channels=3,
                tile_size=model_config['tile_size'],
                memory_requirement_mb=model_config['memory_mb'],
                supports_gpu=True
            )
        
        # OpenCV fallback model
        self.model_info['opencv_cubic'] = ModelInfo(
            name='opencv_cubic',
            path='builtin',
            scale_factor=4,
            input_channels=3,
            tile_size=1024,
            memory_requirement_mb=50,
            supports_gpu=False
        )
        
        logger.info(f"Registered {len(self.model_info)} AI models")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model names.
        
        Returns:
            List of model names
        """
        available = []
        for name, info in self.model_info.items():
            if name == 'opencv_cubic' or Path(info.path).exists():
                available.append(name)
        return available
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelInfo object or None if not found
        """
        return self.model_info.get(model_name)
    
    def load_model(self, model_name: str, use_gpu: bool = True) -> bool:
        """
        Load an AI model for use.
        
        Args:
            model_name: Name of the model to load
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            True if model loaded successfully
        """
        try:
            # Check if model is already loaded
            if self.current_model_name == model_name and self.current_model is not None:
                logger.info(f"Model {model_name} already loaded")
                return True
            
            # Check if model exists in registry
            if model_name not in self.model_info:
                logger.error(f"Unknown model: {model_name}")
                return False
            
            model_info = self.model_info[model_name]
            
            # Check if model file exists (except for builtin models)
            if model_info.path != 'builtin' and not Path(model_info.path).exists():
                logger.error(f"Model file not found: {model_info.path}")
                return False
            
            # Check if model is in cache
            if model_name in self.model_cache:
                self.current_model = self.model_cache[model_name]
                self.current_model_name = model_name
                logger.info(f"Loaded model {model_name} from cache")
                return True
            
            # Create model instance
            model = self._create_model_instance(model_name)
            if model is None:
                return False
            
            # Load the model
            success = model.load(model_info.path, use_gpu)
            if not success:
                logger.error(f"Failed to load model {model_name}")
                return False
            
            # Add to cache (remove oldest if cache is full)
            if len(self.model_cache) >= self.cache_size:
                self._evict_oldest_model()
            
            self.model_cache[model_name] = model
            self.current_model = model
            self.current_model_name = model_name
            
            logger.info(f"Successfully loaded model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Perform prediction with current model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Upscaled image
        """
        if self.current_model is None:
            raise RuntimeError("No model loaded")
        
        return self.current_model.predict(image)
    
    def predict_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Perform batch prediction with current model.
        
        Args:
            images: List of input images
            
        Returns:
            List of upscaled images
        """
        if self.current_model is None:
            raise RuntimeError("No model loaded")
        
        return self.current_model.predict_batch(images)
    
    def get_recommended_tile_config(self, video_resolution: Tuple[int, int]) -> Optional[TileConfiguration]:
        """
        Get recommended tile configuration for current model.
        
        Args:
            video_resolution: Input video resolution (width, height)
            
        Returns:
            Recommended TileConfiguration or None
        """
        if self.current_model_name is None:
            return None
        
        model_info = self.model_info[self.current_model_name]
        
        # Use memory manager to calculate optimal configuration
        from .memory_manager import ConservativeMemoryManager
        memory_manager = ConservativeMemoryManager()
        
        return memory_manager.calculate_optimal_tile_size(
            video_resolution, 
            model_info.memory_requirement_mb
        )
    
    def unload_model(self, model_name: Optional[str] = None) -> None:
        """
        Unload a specific model or current model.
        
        Args:
            model_name: Name of model to unload, or None for current model
        """
        try:
            if model_name is None:
                model_name = self.current_model_name
            
            if model_name is None:
                return
            
            if model_name in self.model_cache:
                model = self.model_cache[model_name]
                model.unload()
                del self.model_cache[model_name]
                logger.info(f"Unloaded model {model_name}")
            
            if model_name == self.current_model_name:
                self.current_model = None
                self.current_model_name = None
                
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        try:
            for model_name in list(self.model_cache.keys()):
                self.unload_model(model_name)
            
            self.model_cache.clear()
            self.current_model = None
            self.current_model_name = None
            
            logger.info("Cleared model cache")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage of all cached models.
        
        Returns:
            Dictionary mapping model names to memory usage in MB
        """
        usage = {}
        for name, model in self.model_cache.items():
            usage[name] = model.get_memory_usage()
        return usage
    
    def auto_select_model(self, video_resolution: Tuple[int, int], 
                         available_memory_mb: int) -> Optional[str]:
        """
        Automatically select best model based on constraints.
        
        Args:
            video_resolution: Input video resolution
            available_memory_mb: Available system memory
            
        Returns:
            Name of recommended model or None
        """
        try:
            available_models = self.get_available_models()
            
            # Filter models by memory requirements
            suitable_models = []
            for model_name in available_models:
                model_info = self.model_info[model_name]
                if model_info.memory_requirement_mb <= available_memory_mb:
                    suitable_models.append((model_name, model_info))
            
            if not suitable_models:
                # Fall back to OpenCV if no AI models fit
                return 'opencv_cubic' if 'opencv_cubic' in available_models else None
            
            # Prefer GPU-capable models if GPU is available
            gpu_available = check_gpu_availability().get('gpu_available', False)
            if gpu_available:
                gpu_models = [(name, info) for name, info in suitable_models if info.supports_gpu]
                if gpu_models:
                    suitable_models = gpu_models
            
            # Select model with highest quality (prefer Real-ESRGAN over OpenCV)
            for name, info in suitable_models:
                if 'real_esrgan' in name:
                    return name
            
            # Fall back to first suitable model
            return suitable_models[0][0] if suitable_models else None
            
        except Exception as e:
            logger.error(f"Error in auto model selection: {e}")
            return 'opencv_cubic'  # Safe fallback
    
    def _create_model_instance(self, model_name: str) -> Optional[AIModel]:
        """Create model instance based on model name."""
        try:
            if 'real_esrgan' in model_name:
                return RealESRGANModel()
            elif model_name == 'opencv_cubic':
                return OpenCVModel()
            else:
                logger.error(f"Unknown model type: {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating model instance {model_name}: {e}")
            return None
    
    def _evict_oldest_model(self) -> None:
        """Evict oldest model from cache."""
        if not self.model_cache:
            return
        
        # Simple LRU: remove first item (oldest)
        oldest_name = next(iter(self.model_cache))
        self.unload_model(oldest_name)


class GPUMemoryMonitor:
    """
    GPU memory monitoring system for strategy-driven model management.
    
    Provides precise GPU memory tracking and dynamic tile size optimization
    to ensure 6GB GPU memory constraint compliance.
    """
    
    def __init__(self, max_gpu_memory_mb: int = 5500):
        """
        Initialize GPU memory monitor.
        
        Args:
            max_gpu_memory_mb: Maximum GPU memory to use (default 5500MB, leaving 500MB buffer)
        """
        self.max_gpu_memory_mb = max_gpu_memory_mb  # Reserve 500MB buffer from 6GB
        logger.info(f"GPU memory monitor initialized with {max_gpu_memory_mb}MB limit")
    
    def check_gpu_memory_available(self) -> int:
        """
        Check available GPU memory in MB with precise tracking.
        
        Returns:
            Available GPU memory in MB, 0 if GPU not available
        """
        try:
            import torch
            if torch.cuda.is_available():
                # Get device properties
                device_props = torch.cuda.get_device_properties(0)
                total_memory = device_props.total_memory
                
                # Get current memory usage
                allocated = torch.cuda.memory_allocated(0)
                cached = torch.cuda.memory_reserved(0)
                
                # Calculate in MB
                total_mb = total_memory // (1024 * 1024)
                allocated_mb = allocated // (1024 * 1024)
                cached_mb = cached // (1024 * 1024)
                used_mb = allocated_mb + cached_mb
                
                # Apply our constraint limit
                effective_total = min(total_mb, self.max_gpu_memory_mb)
                available = max(0, effective_total - used_mb)
                
                logger.debug(f"GPU memory: total={total_mb}MB, allocated={allocated_mb}MB, "
                           f"cached={cached_mb}MB, available={available}MB")
                
                return available
            return 0
        except ImportError:
            logger.warning("PyTorch not available for GPU memory monitoring")
            return 0
        except Exception as e:
            logger.error(f"Error checking GPU memory: {e}")
            return 0
    
    def get_optimal_tile_size(self, base_tile_size: int, model_memory_mb: int) -> int:
        """
        Get optimal tile size based on available GPU memory.
        
        Args:
            base_tile_size: Base tile size to adjust
            model_memory_mb: Memory requirement of the AI model
            
        Returns:
            Optimized tile size (64-512px range)
        """
        available = self.check_gpu_memory_available()
        
        # Calculate safe memory threshold
        safe_memory = available - 200  # 200MB safety buffer
        
        if safe_memory >= model_memory_mb + 3000:  # 3GB余量
            optimal_size = min(512, base_tile_size * 2)
        elif safe_memory >= model_memory_mb + 1000:  # 1GB余量
            optimal_size = base_tile_size
        elif safe_memory >= model_memory_mb + 500:   # 500MB余量
            optimal_size = max(128, base_tile_size // 2)
        else:
            optimal_size = 64  # Minimum safe tile size
        
        logger.debug(f"Optimal tile size: {optimal_size}px (available: {available}MB, "
                    f"model: {model_memory_mb}MB)")
        
        return optimal_size
    
    def force_gpu_memory_cleanup(self) -> None:
        """Force GPU memory cleanup and synchronization with verification."""
        try:
            import torch
            if torch.cuda.is_available():
                # Record memory before cleanup
                before_cleanup = torch.cuda.memory_allocated(0) // (1024 * 1024)
                
                # Perform cleanup
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Verify cleanup
                after_cleanup = torch.cuda.memory_allocated(0) // (1024 * 1024)
                freed_mb = before_cleanup - after_cleanup
                
                logger.debug(f"GPU memory cleanup: freed {freed_mb}MB "
                           f"(before: {before_cleanup}MB, after: {after_cleanup}MB)")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"GPU memory cleanup failed: {e}")
    
    def get_memory_stats(self) -> Dict[str, int]:
        """
        Get detailed GPU memory statistics.
        
        Returns:
            Dictionary with memory statistics in MB
        """
        try:
            import torch
            if torch.cuda.is_available():
                device_props = torch.cuda.get_device_properties(0)
                total_memory = device_props.total_memory // (1024 * 1024)
                allocated = torch.cuda.memory_allocated(0) // (1024 * 1024)
                cached = torch.cuda.memory_reserved(0) // (1024 * 1024)
                
                return {
                    'total_mb': total_memory,
                    'allocated_mb': allocated,
                    'cached_mb': cached,
                    'used_mb': allocated + cached,
                    'available_mb': self.check_gpu_memory_available(),
                    'limit_mb': self.max_gpu_memory_mb
                }
            return {'error': 'GPU not available'}
        except ImportError:
            return {'error': 'PyTorch not available'}
        except Exception as e:
            return {'error': str(e)}


class StrategyDrivenModelManager(AIModelManager):
    """
    Strategy-driven AI model manager for theater enhancement.
    
    Extends AIModelManager with strategy-driven model scheduling,
    strict GPU memory management, and masked prediction capabilities.
    Ensures maximum 1 model loaded at any time to comply with 6GB GPU constraint.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize strategy-driven model manager."""
        super().__init__(*args, **kwargs)
        self.current_strategy: Optional[EnhancementStrategy] = None
        self.gpu_memory_monitor = GPUMemoryMonitor()
        
        # Override cache size to enforce single model constraint
        self.cache_size = 1
        
        logger.info("Strategy-driven model manager initialized with single model constraint")
    
    def set_strategy(self, strategy: EnhancementStrategy) -> None:
        """
        Set current enhancement strategy.
        
        Args:
            strategy: Enhancement strategy configuration
        """
        self.current_strategy = strategy
        logger.info(f"Strategy set: GAN={strategy.gan_policy.global_allowed}, "
                   f"temporal_lock={strategy.temporal_strategy.background_lock}")
    
    def execute_strategy_phase(self, phase: str) -> bool:
        """
        Execute strategy phase with appropriate model loading.
        
        Args:
            phase: Processing phase ("structure_sr", "gan_enhance", "temporal_lock")
            
        Returns:
            True if phase executed successfully
        """
        if not self.current_strategy:
            logger.warning("No strategy set, using default model selection")
            return self.load_model(self.get_available_models()[0] if self.get_available_models() else 'opencv_cubic')
        
        # Determine required model for this phase
        required_model = self._get_required_model_for_phase(phase)
        
        # Switch model if needed
        if required_model != self.current_model_name:
            success = self._switch_model(required_model)
            if not success and required_model is not None:
                logger.warning(f"Failed to load {required_model}, trying fallback")
                return self._switch_model('opencv_cubic')
            return success
        
        return True
    
    def predict_masked(self, image: np.ndarray, mask: np.ndarray) -> List[Dict]:
        """
        Perform AI enhancement prediction on specified regions only.
        
        This method extracts connected regions from the mask and applies
        AI enhancement only to those regions, providing precise control
        over where GAN enhancement is applied.
        
        Args:
            image: Input image as numpy array (H, W, C)
            mask: Boolean mask, True indicates regions to enhance
            
        Returns:
            List of enhanced region dictionaries with keys:
            - 'region': Enhanced region as numpy array
            - 'mask': Region mask as boolean array  
            - 'bbox': Bounding box as (x, y, w, h) tuple
        """
        if self.current_model is None:
            raise RuntimeError("No model loaded for masked prediction")
        
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image and mask shape mismatch: {image.shape[:2]} vs {mask.shape[:2]}")
        
        enhanced_regions = []
        
        try:
            # Convert mask to uint8 for contour detection
            mask_uint8 = mask.astype(np.uint8) * 255
            
            # Find connected components in mask
            contours, _ = cv2.findContours(
                mask_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            logger.debug(f"Found {len(contours)} regions in mask for enhancement")
            
            for i, contour in enumerate(contours):
                try:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Skip regions that are too small (less than 32x32)
                    if w < 32 or h < 32:
                        logger.debug(f"Skipping small region {i}: {w}x{h}")
                        continue
                    
                    # Skip regions that are too large (memory constraint)
                    max_region_size = 1024  # Maximum region size
                    if w > max_region_size or h > max_region_size:
                        logger.warning(f"Skipping large region {i}: {w}x{h} (exceeds {max_region_size}px)")
                        continue
                    
                    # Extract region and region mask
                    region = image[y:y+h, x:x+w].copy()
                    region_mask = mask[y:y+h, x:x+w].copy()
                    
                    # Verify region has enhancement area
                    enhancement_ratio = np.sum(region_mask) / region_mask.size
                    if enhancement_ratio < 0.1:  # Less than 10% needs enhancement
                        logger.debug(f"Skipping region {i}: low enhancement ratio {enhancement_ratio:.2f}")
                        continue
                    
                    # Apply AI enhancement to the region
                    enhanced_region = self.current_model.predict(region)
                    
                    # Validate enhancement result
                    if enhanced_region is None or enhanced_region.shape[:2] != (h * 4, w * 4):
                        logger.warning(f"Invalid enhancement result for region {i}, using original")
                        enhanced_region = region
                    
                    enhanced_regions.append({
                        'region': enhanced_region,
                        'mask': region_mask,
                        'bbox': (x, y, w, h),
                        'enhancement_ratio': enhancement_ratio
                    })
                    
                    logger.debug(f"Enhanced region {i}: bbox=({x},{y},{w},{h}), "
                               f"enhancement_ratio={enhancement_ratio:.2f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to enhance region {i} at ({x},{y},{w},{h}): {e}")
                    # Add original region as fallback
                    try:
                        region = image[y:y+h, x:x+w].copy()
                        region_mask = mask[y:y+h, x:x+w].copy()
                        enhanced_regions.append({
                            'region': region,
                            'mask': region_mask,
                            'bbox': (x, y, w, h),
                            'enhancement_ratio': 0.0  # Mark as not enhanced
                        })
                    except:
                        pass  # Skip this region entirely if fallback fails
            
            logger.info(f"Successfully enhanced {len(enhanced_regions)} regions from mask")
            return enhanced_regions
            
        except Exception as e:
            logger.error(f"Masked prediction failed: {e}")
            # Return empty list on complete failure
            return []
    
    def _get_required_model_for_phase(self, phase: str) -> Optional[str]:
        """
        Determine required model based on phase and current strategy.
        
        Args:
            phase: Processing phase name
            
        Returns:
            Model name or None if no model needed
        """
        if not self.current_strategy:
            return self.get_available_models()[0] if self.get_available_models() else 'opencv_cubic'
        
        if phase == "structure_sr":
            # Structure reconstruction uses traditional methods
            return "opencv_cubic"
        elif phase == "gan_enhance":
            if not self.current_strategy.gan_policy.global_allowed:
                return None  # Skip GAN phase
            
            # Select GAN model based on strength
            strength = self.current_strategy.gan_policy.strength
            if strength in ["medium", "strong"]:
                return "real_esrgan_x4" if "real_esrgan_x4" in self.get_available_models() else "opencv_cubic"
            else:  # weak
                return "real_esrgan_x2" if "real_esrgan_x2" in self.get_available_models() else "opencv_cubic"
        elif phase == "temporal_lock":
            # Temporal locking doesn't need AI models
            return None
        
        return None
    
    def _switch_model(self, model_name: Optional[str]) -> bool:
        """
        Safely switch to a different model with proper memory management.
        Ensures strict compliance with single model constraint.
        
        Args:
            model_name: Name of model to load, None to unload current
            
        Returns:
            True if switch successful
        """
        # Always unload current model first to ensure single model constraint
        if self.current_model:
            logger.debug(f"Unloading current model: {self.current_model_name}")
            self.current_model.unload()
            self.current_model = None
            self.current_model_name = None
            
            # Clear from cache to enforce single model constraint
            if self.current_model_name in self.model_cache:
                del self.model_cache[self.current_model_name]
            
            # Force GPU memory cleanup with synchronization
            self.gpu_memory_monitor.force_gpu_memory_cleanup()
            
            # Verify memory cleanup
            available_memory = self.gpu_memory_monitor.check_gpu_memory_available()
            logger.debug(f"GPU memory after cleanup: {available_memory}MB available")
        
        # Load new model if specified
        if model_name:
            logger.debug(f"Loading new model: {model_name}")
            
            # Check memory availability before loading
            if not self._check_memory_for_model(model_name):
                logger.error(f"Insufficient memory for model: {model_name}")
                return False
            
            success = self.load_model(model_name, use_gpu=True)
            if success:
                logger.info(f"Successfully switched to model: {model_name}")
                
                # Verify single model constraint
                if len(self.model_cache) > 1:
                    logger.warning("Multiple models in cache, enforcing single model constraint")
                    self._enforce_single_model_constraint()
            else:
                logger.error(f"Failed to switch to model: {model_name}")
            return success
        
        return True  # Successfully unloaded
    
    def _check_memory_for_model(self, model_name: str) -> bool:
        """
        Check if sufficient memory is available for loading a model.
        
        Args:
            model_name: Name of model to check
            
        Returns:
            True if sufficient memory available
        """
        if model_name not in self.model_info:
            return False
        
        model_info = self.model_info[model_name]
        available_memory = self.gpu_memory_monitor.check_gpu_memory_available()
        
        # Add safety buffer of 200MB
        required_memory = model_info.memory_requirement_mb + 200
        
        if available_memory < required_memory:
            logger.warning(f"Insufficient GPU memory: need {required_memory}MB, "
                          f"available {available_memory}MB")
            return False
        
        return True
    
    def _enforce_single_model_constraint(self) -> None:
        """Enforce single model constraint by clearing excess models from cache."""
        while len(self.model_cache) > 1:
            # Remove oldest model (first in dict)
            oldest_name = next(iter(self.model_cache))
            if oldest_name != self.current_model_name:
                logger.debug(f"Removing excess model from cache: {oldest_name}")
                self.unload_model(oldest_name)
            else:
                # If current model is oldest, remove second oldest
                model_names = list(self.model_cache.keys())
                if len(model_names) > 1:
                    self.unload_model(model_names[1])
                break
    
    def get_memory_usage_detailed(self) -> Dict[str, Any]:
        """
        Get detailed memory usage information.
        
        Returns:
            Dictionary with CPU and GPU memory usage details
        """
        usage = self.get_memory_usage()  # Get base CPU memory usage
        
        # Add GPU memory information
        gpu_available = self.gpu_memory_monitor.check_gpu_memory_available()
        
        usage.update({
            'gpu_memory_available_mb': gpu_available,
            'gpu_memory_limit_mb': self.gpu_memory_monitor.max_gpu_memory_mb,
            'current_strategy': self.current_strategy.strategy_version if self.current_strategy else None,
            'loaded_model': self.current_model_name
        })
        
        return usage
    
    def optimize_for_strategy(self) -> None:
        """Optimize model manager settings based on current strategy."""
        if not self.current_strategy:
            return
        
        memory_config = self.current_strategy.memory_policy
        
        # Update cache size based on memory policy
        if memory_config.max_model_loaded == 1:
            # Ensure only one model in cache
            while len(self.model_cache) > 1:
                self._evict_oldest_model()
        
        logger.info(f"Model manager optimized for strategy: "
                   f"max_models={memory_config.max_model_loaded}, "
                   f"tile_size={memory_config.tile_size}")
    
    def validate_memory_constraints(self) -> bool:
        """
        Validate that current memory usage meets strategy constraints.
        
        Returns:
            True if memory constraints are satisfied
        """
        if not self.current_strategy:
            return True
        
        try:
            gpu_available = self.gpu_memory_monitor.check_gpu_memory_available()
            memory_config = self.current_strategy.memory_policy
            
            # Check if we have enough memory for the configured tile size
            estimated_usage = self._estimate_memory_usage(memory_config.tile_size)
            
            if gpu_available < estimated_usage:
                logger.warning(f"Insufficient GPU memory: available={gpu_available}MB, "
                              f"required={estimated_usage}MB")
                return False
            
            # Validate single model constraint
            if len(self.model_cache) > memory_config.max_model_loaded:
                logger.warning(f"Too many models loaded: {len(self.model_cache)} > "
                              f"{memory_config.max_model_loaded}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Memory constraint validation failed: {e}")
            return False
    
    def _estimate_memory_usage(self, tile_size: int) -> int:
        """
        Estimate memory usage for given tile size with improved accuracy.
        
        Args:
            tile_size: Tile size in pixels
            
        Returns:
            Estimated memory usage in MB
        """
        try:
            # Calculate base memory for tile processing
            # Formula: tile_size^2 * channels * bytes_per_pixel * scale_factor * processing_overhead
            pixels = tile_size * tile_size
            channels = 3  # RGB
            bytes_per_pixel = 4  # float32
            scale_factor = 4  # 4x upscaling
            processing_overhead = 3  # Input + intermediate + output buffers
            
            tile_memory = (pixels * channels * bytes_per_pixel * scale_factor * processing_overhead) // (1024 * 1024)
            
            # Add model memory if loaded
            model_memory = 0
            if self.current_model_name and self.current_model_name in self.model_info:
                model_memory = self.model_info[self.current_model_name].memory_requirement_mb
            
            # Add safety buffer
            safety_buffer = 200  # 200MB safety buffer
            
            total_estimated = tile_memory + model_memory + safety_buffer
            
            logger.debug(f"Memory estimation: tile={tile_memory}MB, model={model_memory}MB, "
                        f"buffer={safety_buffer}MB, total={total_estimated}MB")
            
            return total_estimated
            
        except Exception as e:
            logger.error(f"Memory estimation failed: {e}")
            return 2000  # Conservative fallback estimate
    
    def get_strategy_optimized_config(self) -> Optional[Dict[str, Any]]:
        """
        Get optimized configuration based on current strategy and available resources.
        
        Returns:
            Dictionary with optimized configuration parameters
        """
        if not self.current_strategy:
            return None
        
        try:
            memory_stats = self.gpu_memory_monitor.get_memory_stats()
            memory_config = self.current_strategy.memory_policy
            
            # Get optimal tile size
            base_tile_size = memory_config.tile_size
            model_memory = 0
            if self.current_model_name and self.current_model_name in self.model_info:
                model_memory = self.model_info[self.current_model_name].memory_requirement_mb
            
            optimal_tile_size = self.gpu_memory_monitor.get_optimal_tile_size(
                base_tile_size, model_memory
            )
            
            # Calculate optimal batch size
            available_memory = memory_stats.get('available_mb', 0)
            if available_memory > 3000:
                optimal_batch_size = min(4, memory_config.batch_size * 2)
            elif available_memory > 1500:
                optimal_batch_size = memory_config.batch_size
            else:
                optimal_batch_size = 1
            
            config = {
                'tile_size': optimal_tile_size,
                'batch_size': optimal_batch_size,
                'use_fp16': memory_config.use_fp16,
                'max_workers': min(memory_config.max_workers, 4),  # Limit CPU workers
                'memory_available_mb': available_memory,
                'model_loaded': self.current_model_name,
                'strategy_version': self.current_strategy.strategy_version
            }
            
            logger.debug(f"Strategy-optimized config: {config}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to generate optimized config: {e}")
            return None
    
    def cleanup_and_validate(self) -> bool:
        """
        Perform cleanup and validate system state.
        
        Returns:
            True if system is in valid state after cleanup
        """
        try:
            # Force memory cleanup
            self.gpu_memory_monitor.force_gpu_memory_cleanup()
            
            # Enforce single model constraint
            self._enforce_single_model_constraint()
            
            # Validate memory constraints
            memory_valid = self.validate_memory_constraints()
            
            # Get current memory stats
            memory_stats = self.gpu_memory_monitor.get_memory_stats()
            
            logger.info(f"Cleanup completed: memory_valid={memory_valid}, "
                       f"available_memory={memory_stats.get('available_mb', 0)}MB, "
                       f"models_loaded={len(self.model_cache)}")
            
            return memory_valid
            
        except Exception as e:
            logger.error(f"Cleanup and validation failed: {e}")
            return False