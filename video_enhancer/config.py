"""
Configuration management system for video enhancement tools and parameters.

This module provides centralized configuration management for tool selection,
processing parameters, and environment setup.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

from .models import ProcessingConfig, ProcessingTool, ProcessingEnvironment


class ConfigManager:
    """Manages configuration for video enhancement processing."""
    
    DEFAULT_CONFIG = {
        "tools": {
            "waifu2x-gui": {
                "models": ["realesrgan-x4plus", "real-cugan", "waifu2x"],
                "default_model": "realesrgan-x4plus",
                "tile_size": 512,
                "gpu_acceleration": True
            },
            "video2x": {
                "models": ["realesrgan", "waifu2x", "anime4k", "srmd"],
                "default_model": "realesrgan",
                "tile_size": 400,
                "gpu_acceleration": True
            },
            "real-esrgan": {
                "models": ["RealESRGAN_x4plus", "RealESRNet_x4plus", "RealESRGAN_x4plus_anime_6B"],
                "default_model": "RealESRGAN_x4plus",
                "tile_size": 0,  # 0 means no tiling
                "gpu_acceleration": True
            }
        },
        "processing": {
            "default_scale_factor": 4,
            "default_denoise_level": 1,
            "max_tile_size": 1024,
            "min_tile_size": 128
        },
        "theater_optimization": {
            "lighting_models": {
                "stage": "realesrgan-x4plus",
                "natural": "real-cugan",
                "mixed": "realesrgan-x4plus"
            },
            "complexity_settings": {
                "simple": {"denoise_level": 0, "tile_size": 512},
                "moderate": {"denoise_level": 1, "tile_size": 400},
                "complex": {"denoise_level": 2, "tile_size": 256}
            }
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default location.
        """
        if config_path is None:
            config_path = os.path.join(os.getcwd(), "config", "video_enhancer.json")
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return self._merge_configs(self.DEFAULT_CONFIG, loaded_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                print("Using default configuration.")
        
        return self.DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge loaded config with defaults."""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def get_tool_config(self, tool: ProcessingTool) -> Dict[str, Any]:
        """Get configuration for a specific tool."""
        return self.config["tools"].get(tool.value, {})
    
    def get_available_models(self, tool: ProcessingTool) -> list:
        """Get available models for a tool."""
        tool_config = self.get_tool_config(tool)
        return tool_config.get("models", [])
    
    def get_default_model(self, tool: ProcessingTool) -> str:
        """Get default model for a tool."""
        tool_config = self.get_tool_config(tool)
        return tool_config.get("default_model", "")
    
    def get_recommended_config(self, 
                             tool: ProcessingTool,
                             lighting_type: str = "mixed",
                             scene_complexity: str = "moderate") -> ProcessingConfig:
        """Get recommended processing configuration based on content analysis.
        
        Args:
            tool: Processing tool to use
            lighting_type: Type of lighting in the video
            scene_complexity: Complexity of the scene
            
        Returns:
            ProcessingConfig with recommended settings
        """
        tool_config = self.get_tool_config(tool)
        theater_config = self.config["theater_optimization"]
        processing_config = self.config["processing"]
        
        # Get model based on lighting type
        model = theater_config["lighting_models"].get(
            lighting_type, 
            self.get_default_model(tool)
        )
        
        # Get settings based on scene complexity
        complexity_settings = theater_config["complexity_settings"].get(
            scene_complexity, 
            theater_config["complexity_settings"]["moderate"]
        )
        
        # Determine environment (prefer Windows native when possible)
        environment = ProcessingEnvironment.WINDOWS
        if tool == ProcessingTool.REAL_ESRGAN:
            environment = ProcessingEnvironment.WSL
        
        return ProcessingConfig(
            tool=tool,
            model=model,
            scale_factor=processing_config["default_scale_factor"],
            tile_size=complexity_settings["tile_size"],
            gpu_acceleration=tool_config.get("gpu_acceleration", True),
            denoise_level=complexity_settings["denoise_level"],
            environment=environment
        )
    
    def update_tool_config(self, tool: ProcessingTool, config_updates: Dict[str, Any]) -> None:
        """Update configuration for a specific tool."""
        if tool.value not in self.config["tools"]:
            self.config["tools"][tool.value] = {}
        
        self.config["tools"][tool.value].update(config_updates)
    
    def export_config(self, processing_config: ProcessingConfig) -> Dict[str, Any]:
        """Export processing configuration to dictionary format."""
        return {
            "tool": processing_config.tool.value,
            "model": processing_config.model,
            "scale_factor": processing_config.scale_factor,
            "tile_size": processing_config.tile_size,
            "gpu_acceleration": processing_config.gpu_acceleration,
            "denoise_level": processing_config.denoise_level,
            "environment": processing_config.environment.value
        }