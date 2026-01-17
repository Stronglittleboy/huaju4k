"""
Enhancement Strategy Planner for Theater Enhancement

This module translates Stage 1 structure features into executable enhancement strategies.
It generates machine-executable strategy JSON without understanding video content.
"""

import logging
import hashlib
from typing import List, Tuple
from datetime import datetime

from ..models.data_models import (
    StructureFeatures, EnhancementStrategy, GANPolicy, 
    TemporalConfig, MemoryConfig, AudioConfig
)

logger = logging.getLogger(__name__)


class EnhancementStrategyPlanner:
    """
    Enhancement strategy planner that translates structure features into execution strategies.
    
    This planner generates machine-executable strategy JSON based on numerical features
    without making subjective judgments about video content.
    """
    
    def __init__(self):
        """Initialize the strategy planner."""
        pass
    
    def generate_strategy(self, features: StructureFeatures) -> EnhancementStrategy:
        """
        Generate complete enhancement strategy based on structure features.
        
        Args:
            features: Structure features from Stage 1 analysis
            
        Returns:
            EnhancementStrategy: Complete executable strategy
        """
        logger.info("Generating enhancement strategy from structure features")
        
        # Generate feature hash for traceability
        features_hash = self._generate_features_hash(features)
        
        # Plan resolution processing path
        resolution_plan = self._plan_resolution_path(features.resolution)
        
        # Calculate GAN policy
        gan_policy = self._calculate_gan_policy(features)
        
        # Generate temporal strategy
        temporal_strategy = self._generate_temporal_strategy(features)
        
        # Generate memory policy
        memory_policy = self._generate_memory_policy(features)
        
        # Generate audio strategy
        audio_strategy = self._generate_audio_strategy(features)
        
        # Create complete strategy
        strategy = EnhancementStrategy(
            resolution_plan=resolution_plan,
            gan_policy=gan_policy,
            temporal_strategy=temporal_strategy,
            memory_policy=memory_policy,
            audio_strategy=audio_strategy,
            strategy_version="1.0",
            generation_timestamp=datetime.now(),
            source_features_hash=features_hash
        )
        
        logger.info(f"Strategy generated: resolution_plan={resolution_plan}, "
                   f"gan_allowed={gan_policy.global_allowed}, "
                   f"temporal_lock={temporal_strategy.background_lock}")
        
        return strategy
    
    def _plan_resolution_path(self, resolution: Tuple[int, int]) -> List[str]:
        """
        Plan resolution processing path based on input resolution.
        
        Args:
            resolution: Input video resolution (width, height)
            
        Returns:
            List of scaling steps: ["x2", "x2"] for 1080p and below, ["x2"] for others
        """
        width, height = resolution
        
        # Rule: 1080p and below use two-step scaling to 4K
        if width <= 1920 and height <= 1080:
            return ["x2", "x2"]  # Two steps to 4K
        else:
            return ["x2"]        # One step to 4K
    
    def _calculate_gan_policy(self, features: StructureFeatures) -> GANPolicy:
        """
        Calculate GAN policy based on highlight ratio and noise level.
        
        Args:
            features: Structure features from analysis
            
        Returns:
            GANPolicy: GAN control configuration
        """
        # Default GAN settings
        gan_allowed = True
        gan_strength = "medium"
        
        # Rule: Disable GAN if too many highlights
        if features.highlight_ratio > 0.2:
            gan_allowed = False
            logger.info(f"GAN disabled due to high highlight ratio: {features.highlight_ratio:.3f}")
        
        # Rule: Reduce GAN strength if too much noise
        elif features.noise_score > 0.25:
            gan_strength = "weak"
            logger.info(f"GAN strength reduced to weak due to noise: {features.noise_score:.3f}")
        
        # Rule: Use strong GAN if rich in edges
        elif features.edge_density > 0.3:
            gan_strength = "strong"
            logger.info(f"GAN strength increased to strong due to edge density: {features.edge_density:.3f}")
        
        return GANPolicy(
            global_allowed=gan_allowed,
            strength=gan_strength,
            highlight_threshold=0.85,
            shadow_threshold=0.15,
            edge_threshold=0.1,
            motion_threshold=0.05
        )
    
    def _generate_temporal_strategy(self, features: StructureFeatures) -> TemporalConfig:
        """
        Generate temporal strategy based on static camera detection and frame changes.
        
        Args:
            features: Structure features from analysis
            
        Returns:
            TemporalConfig: Temporal processing configuration
        """
        # Rule: Enable background lock for static camera with low motion
        if features.is_static_camera and features.frame_diff_mean < 0.02:
            background_lock = True
            temporal_strength = "high"
            logger.info("Temporal lock enabled: static camera detected")
        else:
            background_lock = False
            temporal_strength = "medium"
            logger.info("Temporal lock disabled: dynamic camera or high motion")
        
        return TemporalConfig(
            background_lock=background_lock,
            strength=temporal_strength,
            motion_threshold=0.05,
            optical_flow_enabled=True,
            smoothing_alpha=0.3
        )
    
    def _generate_memory_policy(self, features: StructureFeatures) -> MemoryConfig:
        """
        Generate memory policy ensuring 6GB GPU memory constraint.
        
        Args:
            features: Structure features from analysis
            
        Returns:
            MemoryConfig: Memory management configuration
        """
        # Base configuration for 6GB GPU constraint
        base_tile_size = 512
        
        # Adjust tile size based on resolution
        width, height = features.resolution
        total_pixels = width * height
        
        # Rule: Reduce tile size for high resolution inputs
        if total_pixels > 1920 * 1080:  # Above 1080p
            tile_size = 256
        elif total_pixels > 1280 * 720:  # Above 720p
            tile_size = 384
        else:
            tile_size = base_tile_size
        
        logger.info(f"Memory policy: tile_size={tile_size} for resolution {features.resolution}")
        
        return MemoryConfig(
            max_model_loaded=1,      # Strict constraint: only 1 model at a time
            tile_size=tile_size,
            batch_size=1,            # Conservative batch size
            use_fp16=True,           # Use half precision to save memory
            max_workers=4            # CPU parallel workers
        )
    
    def _generate_audio_strategy(self, features: StructureFeatures) -> AudioConfig:
        """
        Generate audio strategy based on video features.
        
        Args:
            features: Structure features from analysis
            
        Returns:
            AudioConfig: Audio processing configuration
        """
        # Default theater preset based on video characteristics
        if features.edge_density > 0.2:
            # High edge density suggests detailed scene - use small theater preset
            theater_preset = "small"
        elif features.highlight_ratio > 0.15:
            # High highlight ratio suggests large space - use large theater preset
            theater_preset = "large"
        else:
            # Default to medium theater preset
            theater_preset = "medium"
        
        logger.info(f"Audio strategy: theater_preset={theater_preset}")
        
        # Get the preset configuration
        audio_config = AudioConfig()
        preset_config = audio_config.get_theater_preset(theater_preset)
        
        # Create audio strategy with recommended preset
        return AudioConfig(
            theater_presets=audio_config.theater_presets,
            sample_rate=48000,
            bitrate="192k",
            channels=2
        )
    
    def _generate_features_hash(self, features: StructureFeatures) -> str:
        """
        Generate hash of structure features for traceability.
        
        Args:
            features: Structure features to hash
            
        Returns:
            Hash string for traceability
        """
        # Create a string representation of key features
        feature_string = (
            f"{features.resolution[0]}x{features.resolution[1]}_"
            f"{features.fps:.2f}fps_"
            f"static:{features.is_static_camera}_"
            f"highlight:{features.highlight_ratio:.3f}_"
            f"edge:{features.edge_density:.3f}_"
            f"motion:{features.frame_diff_mean:.3f}_"
            f"noise:{features.noise_score:.3f}"
        )
        
        # Generate MD5 hash
        return hashlib.md5(feature_string.encode()).hexdigest()[:16]