"""
Enhancement Strategy Planner for Theater Enhancement

Translates Stage 1 structure features into executable enhancement strategies,
including VRAM-aware model selection and scene-level decisions.
"""

import logging
import hashlib
from typing import List, Tuple
from datetime import datetime

from ..models.data_models import (
    StructureFeatures, SceneSegment, EnhancementStrategy,
    GANPolicy, TemporalConfig, MemoryConfig, AudioConfig
)

logger = logging.getLogger(__name__)


class EnhancementStrategyPlanner:
    """Translates StructureFeatures into an EnhancementStrategy."""

    def __init__(self):
        pass

    def generate_strategy(
        self,
        features: StructureFeatures,
        quality: str = "standard",
        codec: str = "h264",
        crf: int = 18,
    ) -> EnhancementStrategy:
        """Generate a complete enhancement strategy from structure features.

        Args:
            features: Structure features from Stage 1 analysis.
            quality: "fast" (lanczos), "standard" (AI + background reuse),
                     or "master" (full-frame AI on every scene).
            codec: Output codec – "h264", "h265", or "prores".
            crf: Constant rate factor for encoding.

        Returns:
            EnhancementStrategy ready for the processing pipeline.
        """
        logger.info("Generating enhancement strategy (quality=%s)", quality)

        features_hash = self._generate_features_hash(features)

        available_vram = self._detect_gpu_memory()
        model_name, tile_size = self._select_model(available_vram, quality)
        denoise_strength = self._map_denoise_strength(features.noise_score)

        resolution_plan = self._plan_resolution_path(features.resolution)
        gan_policy = self._calculate_gan_policy(features)
        temporal_strategy = self._generate_temporal_strategy(features)
        memory_policy = self._generate_memory_policy(features)
        audio_strategy = self._generate_audio_strategy(features)

        strategy = EnhancementStrategy(
            # legacy fields
            resolution_plan=resolution_plan,
            gan_policy=gan_policy,
            temporal_strategy=temporal_strategy,
            memory_policy=memory_policy,
            audio_strategy=audio_strategy,
            strategy_version="2.0",
            generation_timestamp=datetime.now(),
            source_features_hash=features_hash,
            # new fields
            model_name=model_name,
            tile_size=tile_size,
            scenes=list(features.scenes),
            denoise_strength=denoise_strength,
            audio_enabled=features.duration > 0,
            codec=codec,
            crf=crf,
        )

        logger.info(
            "Strategy generated: model=%s tile=%d denoise=%s scenes=%d codec=%s crf=%d",
            model_name, tile_size, denoise_strength, len(strategy.scenes), codec, crf,
        )

        return strategy

    # ------------------------------------------------------------------
    # VRAM / model selection
    # ------------------------------------------------------------------

    def _detect_gpu_memory(self) -> float:
        """Return available GPU memory in GB (0.0 if no CUDA GPU)."""
        try:
            import torch
            if not torch.cuda.is_available():
                return 0.0
            free, _total = torch.cuda.mem_get_info(torch.cuda.current_device())
            vram_gb = free / (1024 ** 3)
            logger.info("Detected %.2f GB available GPU memory", vram_gb)
            return vram_gb
        except Exception:
            logger.info("No CUDA GPU detected, falling back to CPU")
            return 0.0

    def _select_model(self, available_vram: float, quality: str) -> Tuple[str, int]:
        """Choose super-resolution model and tile size.

        Returns:
            (model_name, tile_size)
        """
        if quality == "fast":
            return ("lanczos", 0)

        if available_vram <= 0:
            logger.info("No GPU available, using lanczos upscaling")
            return ("lanczos", 0)

        if available_vram >= 3.5:
            return ("x4plus", 384)

        return ("x2plus", 384)

    # ------------------------------------------------------------------
    # Denoise mapping
    # ------------------------------------------------------------------

    def _map_denoise_strength(self, noise_score: float) -> str:
        if noise_score > 0.3:
            return "high"
        if noise_score > 0.15:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Legacy sub-strategies (kept for backward compat)
    # ------------------------------------------------------------------

    def _plan_resolution_path(self, resolution: Tuple[int, int]) -> List[str]:
        width, height = resolution
        if width <= 1920 and height <= 1080:
            return ["x2", "x2"]
        return ["x2"]

    def _calculate_gan_policy(self, features: StructureFeatures) -> GANPolicy:
        gan_allowed = True
        gan_strength = "medium"

        if features.highlight_ratio > 0.2:
            gan_allowed = False
        elif features.noise_score > 0.25:
            gan_strength = "weak"
        elif features.edge_density > 0.3:
            gan_strength = "strong"

        return GANPolicy(
            global_allowed=gan_allowed,
            strength=gan_strength,
            highlight_threshold=0.85,
            shadow_threshold=0.15,
            edge_threshold=0.1,
            motion_threshold=0.05,
        )

    def _generate_temporal_strategy(self, features: StructureFeatures) -> TemporalConfig:
        if features.is_static_camera and features.frame_diff_mean < 0.02:
            return TemporalConfig(
                background_lock=True, strength="high",
                motion_threshold=0.05, optical_flow_enabled=True, smoothing_alpha=0.3,
            )
        return TemporalConfig(
            background_lock=False, strength="medium",
            motion_threshold=0.05, optical_flow_enabled=True, smoothing_alpha=0.3,
        )

    def _generate_memory_policy(self, features: StructureFeatures) -> MemoryConfig:
        width, height = features.resolution
        total_pixels = width * height

        if total_pixels > 1920 * 1080:
            tile_size = 256
        elif total_pixels > 1280 * 720:
            tile_size = 384
        else:
            tile_size = 512

        return MemoryConfig(
            max_model_loaded=1, tile_size=tile_size,
            batch_size=1, use_fp16=True, max_workers=4,
        )

    def _generate_audio_strategy(self, features: StructureFeatures) -> AudioConfig:
        if features.edge_density > 0.2:
            preset = "small"
        elif features.highlight_ratio > 0.15:
            preset = "large"
        else:
            preset = "medium"

        base = AudioConfig()
        base.get_theater_preset(preset)
        return AudioConfig(
            theater_presets=base.theater_presets,
            sample_rate=48000, bitrate="192k", channels=2,
        )

    def _generate_features_hash(self, features: StructureFeatures) -> str:
        feature_string = (
            f"{features.resolution[0]}x{features.resolution[1]}_"
            f"{features.fps:.2f}fps_"
            f"static:{features.is_static_camera}_"
            f"highlight:{features.highlight_ratio:.3f}_"
            f"edge:{features.edge_density:.3f}_"
            f"motion:{features.frame_diff_mean:.3f}_"
            f"noise:{features.noise_score:.3f}"
        )
        return hashlib.md5(feature_string.encode()).hexdigest()[:16]
