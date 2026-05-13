"""
Unified configuration manager for huaju4k.

Provides a single entry point for loading / saving / querying configuration.
Replaces the previous 7-file config system with one flat module.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..models.config_models import HuaJu4KConfig, PresetConfig, SystemConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "video": {
        "ai_model": "real_esrgan",
        "model_path": "./models/RealESRGAN_x4plus.pth",
        "quality_presets": {
            "fast":     {"tile_size": 512, "batch_size": 4, "denoise_strength": 0.5, "overlap_pixels": 16},
            "balanced": {"tile_size": 768, "batch_size": 2, "denoise_strength": 0.7, "overlap_pixels": 32},
            "high":     {"tile_size": 1024, "batch_size": 1, "denoise_strength": 0.9, "overlap_pixels": 64},
        },
        "output": {"format": "mp4", "codec": "h264", "crf": 18, "preset": "slow"},
    },
    "audio": {
        "theater_presets": {
            "small":  {"reverb_reduction": 0.8, "dialogue_boost": 6.0, "noise_reduction": 0.7},
            "medium": {"reverb_reduction": 0.6, "dialogue_boost": 4.0, "noise_reduction": 0.5},
            "large":  {"reverb_reduction": 0.4, "dialogue_boost": 2.0, "noise_reduction": 0.3},
        },
        "sample_rate": 48000,
        "bitrate": "192k",
    },
    "performance": {
        "use_gpu": True,
        "gpu_id": 0,
        "max_memory_usage": 0.7,
        "checkpoint_interval": 500,
    },
}

DEFAULT_PRESETS: Dict[str, Dict[str, Any]] = {
    "theater_small_fast": {
        "name": "Small Theater - Fast",
        "description": "小剧场快速处理",
        "theater_size": "small",
        "quality_level": "fast",
        "target_resolution": "3840x2160",
        "denoise_strength": 0.5,
        "dialogue_boost": 6.0,
        "noise_reduction": 0.7,
        "reverb_reduction": 0.8,
        "tile_size": 512,
        "batch_size": 4,
        "memory_usage": 0.6,
    },
    "theater_medium_balanced": {
        "name": "Medium Theater - Balanced",
        "description": "中型剧场均衡处理",
        "theater_size": "medium",
        "quality_level": "balanced",
        "target_resolution": "3840x2160",
        "denoise_strength": 0.7,
        "dialogue_boost": 4.0,
        "noise_reduction": 0.5,
        "reverb_reduction": 0.6,
        "tile_size": 768,
        "batch_size": 2,
        "memory_usage": 0.7,
    },
    "theater_large_high": {
        "name": "Large Theater - High Quality",
        "description": "大剧场高质量处理",
        "theater_size": "large",
        "quality_level": "high",
        "target_resolution": "3840x2160",
        "denoise_strength": 0.9,
        "dialogue_boost": 2.0,
        "noise_reduction": 0.3,
        "reverb_reduction": 0.4,
        "tile_size": 1024,
        "batch_size": 1,
        "memory_usage": 0.85,
    },
}


class ConfigManager:
    """统一配置管理器。"""

    def __init__(self, config_path: Optional[str] = None):
        self._config: Dict[str, Any] = DEFAULT_CONFIG.copy()
        self._config_path = Path(config_path) if config_path else None
        if self._config_path and self._config_path.exists():
            self._load_from_file(self._config_path)

    # -- 读 ----------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """点分隔键读取，如 ``video.output.codec``。"""
        parts = key.split(".")
        node: Any = self._config
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def get_config(self) -> Dict[str, Any]:
        return self._config

    def get_preset(self, name: str) -> PresetConfig:
        preset_dict = DEFAULT_PRESETS.get(name)
        if preset_dict is None:
            raise KeyError(f"Preset '{name}' not found. Available: {list(DEFAULT_PRESETS)}")
        return PresetConfig(**preset_dict)

    def list_presets(self) -> list:
        return list(DEFAULT_PRESETS.keys())

    # -- 写 ----------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        node = self._config
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def save(self, path: Optional[str] = None) -> None:
        save_path = Path(path) if path else self._config_path
        if save_path is None:
            raise ValueError("No path specified for saving config")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(self._config, indent=2, ensure_ascii=False))
        logger.info("Config saved to %s", save_path)

    # -- 内部 ---------------------------------------------------------------

    def _load_from_file(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._deep_update(self._config, data)
            logger.info("Config loaded from %s", path)
        except Exception as e:
            logger.warning("Failed to load config from %s: %s", path, e)

    @staticmethod
    def _deep_update(base: Dict, updates: Dict) -> None:
        for k, v in updates.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                ConfigManager._deep_update(base[k], v)
            else:
                base[k] = v


# Legacy alias
SimpleConfigManager = ConfigManager
ConfigurationManager = ConfigManager
