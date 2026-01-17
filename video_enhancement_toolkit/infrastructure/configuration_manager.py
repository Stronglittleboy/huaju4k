"""
Configuration Management System

Implements JSON-based configuration schema, validation, and preset management.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import jsonschema
from jsonschema import validate, ValidationError

from .interfaces import IConfigurationManager
from .models import ConfigurationData, LoggingConfig, LogLevel


class ConfigurationManager(IConfigurationManager):
    """Implementation of configuration management system."""
    
    # JSON Schema for configuration validation
    CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "video_config": {
                "type": "object",
                "properties": {
                    "input_path": {"type": "string"},
                    "output_path": {"type": "string"},
                    "target_resolution": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2
                    },
                    "ai_model": {"type": "string"},
                    "tile_size": {"type": "integer", "minimum": 64, "maximum": 2048},
                    "batch_size": {"type": "integer", "minimum": 1, "maximum": 32},
                    "gpu_acceleration": {"type": "boolean"}
                },
                "required": ["ai_model", "tile_size", "batch_size"]
            },
            "audio_config": {
                "type": "object",
                "properties": {
                    "noise_reduction_strength": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "dialogue_enhancement": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "dynamic_range_target": {"type": "number", "minimum": -60.0, "maximum": 0.0},
                    "preserve_naturalness": {"type": "boolean"},
                    "theater_preset": {"type": "string", "enum": ["small", "medium", "large"]}
                }
            },
            "performance_config": {
                "type": "object",
                "properties": {
                    "cpu_threads": {"type": "integer", "minimum": 1, "maximum": 64},
                    "gpu_memory_limit": {"type": "number", "minimum": 0.1, "maximum": 64.0},
                    "memory_optimization": {"type": "boolean"},
                    "parallel_processing": {"type": "boolean"}
                }
            },
            "logging_config": {
                "type": "object",
                "properties": {
                    "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                    "log_file_path": {"type": ["string", "null"]},
                    "console_output": {"type": "boolean"},
                    "json_format": {"type": "boolean"},
                    "max_file_size": {"type": "integer", "minimum": 1024},
                    "backup_count": {"type": "integer", "minimum": 0}
                }
            },
            "presets": {
                "type": "object",
                "additionalProperties": {
                    "type": "object"
                }
            }
        }
    }
    
    def __init__(self, config_dir: str = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory for configuration files
        """
        if config_dir is None:
            config_dir = os.path.join(os.getcwd(), "config")
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / "video_enhancement_toolkit.json"
        self.presets_file = self.config_dir / "presets.json"
        
        self._config_data = None
        self._presets = None
    
    def load_configuration(self, config_path: Optional[str] = None) -> ConfigurationData:
        """Load configuration from file.
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            ConfigurationData object with loaded configuration
        """
        if config_path:
            config_file = Path(config_path)
        else:
            config_file = self.config_file
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                
                # Validate configuration
                if not self.validate_configuration(config_dict):
                    raise ValueError("Configuration validation failed")
                
                self._config_data = ConfigurationData(
                    video_config=config_dict.get("video_config", {}),
                    audio_config=config_dict.get("audio_config", {}),
                    performance_config=config_dict.get("performance_config", {}),
                    logging_config=config_dict.get("logging_config", {}),
                    presets=config_dict.get("presets", {})
                )
                
            except (json.JSONDecodeError, IOError, ValueError) as e:
                print(f"Warning: Could not load config from {config_file}: {e}")
                print("Using default configuration.")
                self._config_data = self._get_default_configuration()
        else:
            self._config_data = self._get_default_configuration()
        
        return self._config_data
    
    def save_configuration(self, config: ConfigurationData, config_path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration data to save
            config_path: Optional path to save configuration
        """
        if config_path:
            config_file = Path(config_path)
        else:
            config_file = self.config_file
        
        config_dict = {
            "video_config": config.video_config,
            "audio_config": config.audio_config,
            "performance_config": config.performance_config,
            "logging_config": config.logging_config,
            "presets": config.presets
        }
        
        # Validate before saving
        if not self.validate_configuration(config_dict):
            raise ValueError("Configuration validation failed")
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        self._config_data = config
    
    def get_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration preset by name.
        
        Args:
            preset_name: Name of the preset
            
        Returns:
            Dictionary with preset configuration or None if not found
        """
        if self._config_data is None:
            self.load_configuration()
        
        # Check built-in theater presets first
        theater_presets = self._config_data.get_theater_presets()
        if preset_name in theater_presets:
            return theater_presets[preset_name]
        
        # Check user-defined presets
        return self._config_data.presets.get(preset_name)
    
    def save_preset(self, preset_name: str, config: Dict[str, Any]) -> None:
        """Save configuration as preset.
        
        Args:
            preset_name: Name for the preset
            config: Configuration data to save as preset
        """
        if self._config_data is None:
            self.load_configuration()
        
        # Validate preset configuration
        if not self._validate_preset_config(config):
            raise ValueError("Preset configuration validation failed")
        
        self._config_data.presets[preset_name] = config
        self.save_configuration(self._config_data)
    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            validate(instance=config, schema=self.CONFIG_SCHEMA)
            return True
        except ValidationError as e:
            print(f"Configuration validation error: {e.message}")
            return False
    
    def get_available_presets(self) -> List[str]:
        """Get list of available preset names.
        
        Returns:
            List of preset names
        """
        if self._config_data is None:
            self.load_configuration()
        
        theater_presets = list(self._config_data.get_theater_presets().keys())
        user_presets = list(self._config_data.presets.keys())
        
        return theater_presets + user_presets
    
    def _get_default_configuration(self) -> ConfigurationData:
        """Get default configuration.
        
        Returns:
            ConfigurationData with default values
        """
        return ConfigurationData(
            video_config={
                "ai_model": "realesrgan-x4plus",
                "tile_size": 512,
                "batch_size": 4,
                "gpu_acceleration": True,
                "target_resolution": [3840, 2160]
            },
            audio_config={
                "noise_reduction_strength": 0.3,
                "dialogue_enhancement": 0.4,
                "dynamic_range_target": -23.0,
                "preserve_naturalness": True,
                "theater_preset": "medium"
            },
            performance_config={
                "cpu_threads": 4,
                "gpu_memory_limit": 4.0,
                "memory_optimization": True,
                "parallel_processing": True
            },
            logging_config={
                "log_level": "INFO",
                "log_file_path": None,
                "console_output": True,
                "json_format": True,
                "max_file_size": 10 * 1024 * 1024,
                "backup_count": 5
            },
            presets={}
        )
    
    def _validate_preset_config(self, config: Dict[str, Any]) -> bool:
        """Validate preset configuration.
        
        Args:
            config: Preset configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Preset should contain at least one of the main config sections
        valid_sections = ["video_config", "audio_config", "performance_config", "logging_config"]
        has_valid_section = any(section in config for section in valid_sections)
        
        if not has_valid_section:
            return False
        
        # Validate each section that exists
        for section, section_config in config.items():
            if section in valid_sections:
                temp_config = {section: section_config}
                if not self.validate_configuration(temp_config):
                    return False
        
        return True