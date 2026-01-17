"""
Configuration management system for huaju4k video enhancement.

This module provides configuration loading, validation, and management
functionality for the application with enhanced YAML support and validation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

# Optional imports with fallbacks
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

from ..models.config_models import HuaJu4KConfig, PresetConfig, SystemConfig
from ..utils.validation_utils import validate_config
from .default_config import get_default_config, get_default_preset
from .yaml_config_loader import YAMLConfigLoader, ConfigValidationError
from .enhanced_preset_manager import PresetManager, PresetValidationError

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Enhanced configuration management system with YAML support and validation.
    
    Provides comprehensive configuration management including loading, saving,
    validation, and preset management with enhanced error handling.
    """
    
    def __init__(self, config_dir: Optional[str] = None, 
                 presets_dir: Optional[str] = None,
                 system_config: Optional[SystemConfig] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory for configuration files (default: ./config)
            presets_dir: Directory for preset files (default: ./presets)
            system_config: System configuration for compatibility checking
        """
        self.config_dir = Path(config_dir or "./config")
        self.presets_dir = Path(presets_dir or "./presets")
        self.system_config = system_config
        
        # Initialize components
        self.yaml_loader = YAMLConfigLoader(strict_validation=True)
        self.preset_manager = PresetManager(self.presets_dir, system_config)
        
        # Configuration state
        self.config_file = self.config_dir / "huaju4k.yaml"
        self._current_config: Optional[HuaJu4KConfig] = None
        
        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"ConfigurationManager initialized")
    
    def load_config(self, config_path: Optional[str] = None, 
                   fallback_to_defaults: bool = True) -> HuaJu4KConfig:
        """
        Load configuration from file with enhanced validation.
        
        Args:
            config_path: Optional path to specific config file
            fallback_to_defaults: Whether to use defaults if file not found
            
        Returns:
            HuaJu4KConfig object
            
        Raises:
            ConfigValidationError: If configuration is invalid
            FileNotFoundError: If file not found and fallback disabled
        """
        try:
            if config_path:
                config_file = Path(config_path)
            else:
                config_file = self.config_file
            
            # Use enhanced YAML loader
            self._current_config = self.yaml_loader.load_config(
                config_file, 
                fallback_to_defaults=fallback_to_defaults
            )
            
            logger.info(f"Configuration loaded successfully from {config_file}")
            return self._current_config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def save_config(self, config: Optional[HuaJu4KConfig] = None, 
                   config_path: Optional[str] = None,
                   create_backup: bool = True) -> None:
        """
        Save configuration to file with backup support.
        
        Args:
            config: Configuration to save (default: current config)
            config_path: Optional path to save to (default: default config file)
            create_backup: Whether to create backup of existing file
            
        Raises:
            ValueError: If no configuration to save
            OSError: If file operations fail
        """
        try:
            config_to_save = config or self._current_config
            if config_to_save is None:
                raise ValueError("No configuration to save")
            
            if config_path:
                save_file = Path(config_path)
            else:
                save_file = self.config_file
            
            # Use enhanced YAML loader for saving
            self.yaml_loader.save_config(config_to_save, save_file, create_backup)
            
            logger.info(f"Configuration saved to {save_file}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def get_config(self) -> HuaJu4KConfig:
        """
        Get current configuration, loading if necessary.
        
        Returns:
            Current HuaJu4KConfig object
        """
        if self._current_config is None:
            self.load_config()
        return self._current_config
    
    def update_config(self, updates: Dict[str, Any], validate: bool = True) -> None:
        """
        Update current configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            validate: Whether to validate updates
            
        Raises:
            ConfigValidationError: If updates are invalid
        """
        try:
            current_config = self.get_config()
            current_dict = current_config.to_dict()
            
            # Apply updates (deep merge)
            self._deep_update(current_dict, updates)
            
            # Validate updated configuration if requested
            if validate:
                validation_result = validate_config(current_dict)
                if not validation_result['valid']:
                    raise ConfigValidationError(
                        f"Invalid configuration updates: {validation_result['errors']}",
                        errors=validation_result['errors'],
                        warnings=validation_result.get('warnings', [])
                    )
            
            # Create new config object
            self._current_config = HuaJu4KConfig.from_dict(current_dict)
            logger.info("Configuration updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            raise
    
    def validate_config_file(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate configuration file without loading it.
        
        Args:
            config_path: Path to configuration file to validate
            
        Returns:
            Dictionary containing validation results
        """
        try:
            if config_path:
                config_file = Path(config_path)
            else:
                config_file = self.config_file
            
            return self.yaml_loader.validate_config_file(config_file)
            
        except Exception as e:
            logger.error(f"Error validating configuration file: {e}")
            return {
                'valid': False,
                'errors': [str(e)],
                'warnings': [],
                'file_exists': False,
                'file_readable': False,
                'yaml_valid': False,
                'structure_valid': False
            }
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        try:
            self._current_config = HuaJu4KConfig.from_dict(get_default_config())
            logger.info("Configuration reset to defaults")
        except Exception as e:
            logger.error(f"Error resetting configuration: {e}")
            raise
    
    def backup_config(self, backup_path: Optional[str] = None) -> str:
        """
        Create backup of current configuration.
        
        Args:
            backup_path: Optional path for backup file
            
        Returns:
            Path to backup file
            
        Raises:
            ValueError: If no configuration to backup
        """
        try:
            if self._current_config is None:
                raise ValueError("No configuration to backup")
            
            if backup_path:
                backup_file = Path(backup_path)
            else:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = self.config_dir / f"huaju4k_backup_{timestamp}.yaml"
            
            # Save backup using YAML loader
            self.yaml_loader.save_config(self._current_config, backup_file, create_backup=False)
            
            logger.info(f"Configuration backed up to {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Error backing up configuration: {e}")
            raise
    
    # Preset management methods (delegate to PresetManager)
    
    def load_preset(self, preset_name: str) -> PresetConfig:
        """Load preset configuration."""
        return self.preset_manager.load_preset(preset_name)
    
    def save_preset(self, preset: PresetConfig, preset_name: Optional[str] = None,
                   overwrite: bool = False) -> str:
        """Save preset configuration."""
        return self.preset_manager.save_preset(preset, preset_name, overwrite)
    
    def create_custom_preset(self, name: str, description: str,
                           base_preset: str = "theater_medium_balanced",
                           overrides: Optional[Dict[str, Any]] = None) -> PresetConfig:
        """Create custom preset with parameter overrides."""
        return self.preset_manager.create_custom_preset(name, description, base_preset, overrides)
    
    def list_presets(self) -> List[str]:
        """List all available presets."""
        return self.preset_manager.list_available_presets()
    
    def get_preset_info(self, preset_name: str) -> Dict[str, Any]:
        """Get detailed preset information."""
        return self.preset_manager.get_preset_info(preset_name)
    
    def delete_preset(self, preset_name: str, force: bool = False) -> bool:
        """Delete preset file."""
        return self.preset_manager.delete_preset(preset_name, force)
    
    def validate_preset(self, preset: PresetConfig) -> Dict[str, Any]:
        """Validate preset configuration."""
        return self.preset_manager.validate_preset(preset)
    
    def export_preset(self, preset_name: str, export_path: Union[str, Path]) -> str:
        """Export preset to specified location."""
        return self.preset_manager.export_preset(preset_name, export_path)
    
    def import_preset(self, import_path: Union[str, Path], 
                     preset_name: Optional[str] = None,
                     overwrite: bool = False) -> str:
        """Import preset from file."""
        return self.preset_manager.import_preset(import_path, preset_name, overwrite)
    
    def apply_preset_overrides(self, preset: PresetConfig, 
                             overrides: Dict[str, Any]) -> PresetConfig:
        """Apply parameter overrides to preset."""
        return self.preset_manager.apply_parameter_overrides(preset, overrides)
    
    # Utility methods
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information and configuration status.
        
        Returns:
            Dictionary containing system and configuration information
        """
        info = {
            'config_dir': str(self.config_dir),
            'presets_dir': str(self.presets_dir),
            'config_file_exists': self.config_file.exists(),
            'yaml_support': HAS_YAML,
            'available_presets': self.list_presets(),
            'current_config_loaded': self._current_config is not None
        }
        
        if self.system_config:
            info['system_config'] = {
                'cpu_cores': self.system_config.cpu_cores,
                'total_memory_mb': self.system_config.total_memory_mb,
                'gpu_available': self.system_config.gpu_available,
                'gpu_memory_mb': self.system_config.gpu_memory_mb
            }
        
        return info
    
    def _deep_update(self, base_dict: Dict[str, Any], 
                    update_dict: Dict[str, Any]) -> None:
        """
        Recursively update nested dictionary.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Updates to apply
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


# Legacy compatibility - keep old class name as alias
ConfigManager = ConfigurationManager