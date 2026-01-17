"""
Enhanced preset management system for huaju4k video enhancement.

This module provides comprehensive preset management with validation,
custom preset creation, parameter override, and compatibility checking.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging

from ..models.config_models import PresetConfig, SystemConfig
from .yaml_config_loader import YAMLConfigLoader, ConfigValidationError
from .default_config import get_default_preset, get_available_presets, validate_preset_compatibility

logger = logging.getLogger(__name__)


class PresetValidationError(Exception):
    """Exception raised when preset validation fails."""
    pass


class PresetManager:
    """
    Enhanced preset management system with validation and compatibility checking.
    
    Provides comprehensive preset management including loading, saving, validation,
    custom preset creation, and parameter override functionality.
    """
    
    def __init__(self, presets_dir: Union[str, Path] = "./presets",
                 system_config: Optional[SystemConfig] = None):
        """
        Initialize preset manager.
        
        Args:
            presets_dir: Directory for preset files
            system_config: System configuration for compatibility checking
        """
        self.presets_dir = Path(presets_dir)
        self.system_config = system_config
        self.yaml_loader = YAMLConfigLoader(strict_validation=True)
        
        # Ensure presets directory exists
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded presets
        self._preset_cache: Dict[str, PresetConfig] = {}
        self._preset_metadata_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.debug(f"PresetManager initialized with directory: {self.presets_dir}")
    
    def load_preset(self, preset_name: str, use_cache: bool = True) -> PresetConfig:
        """
        Load preset configuration with caching and validation.
        
        Args:
            preset_name: Name of preset to load
            use_cache: Whether to use cached preset if available
            
        Returns:
            PresetConfig object
            
        Raises:
            FileNotFoundError: If preset not found
            PresetValidationError: If preset is invalid
        """
        try:
            # Check cache first
            if use_cache and preset_name in self._preset_cache:
                logger.debug(f"Returning cached preset: {preset_name}")
                return self._preset_cache[preset_name]
            
            preset = None
            
            # Try to load from file first
            preset_file = self.presets_dir / f"{preset_name}.yaml"
            if preset_file.exists():
                try:
                    preset = self.yaml_loader.load_preset(preset_file)
                    logger.info(f"Loaded preset '{preset_name}' from file")
                except ConfigValidationError as e:
                    logger.error(f"Invalid preset file '{preset_name}': {e}")
                    raise PresetValidationError(f"Invalid preset file: {e}")
            
            # Fall back to default presets
            if preset is None:
                try:
                    preset_dict = get_default_preset(preset_name)
                    preset = PresetConfig(**preset_dict)
                    logger.info(f"Loaded default preset: {preset_name}")
                except KeyError:
                    available = self.list_available_presets()
                    raise FileNotFoundError(
                        f"Preset '{preset_name}' not found. Available presets: {available}"
                    )
            
            # Validate preset compatibility
            if self.system_config:
                compatibility = self._check_preset_compatibility(preset)
                if not compatibility['compatible']:
                    logger.warning(f"Preset '{preset_name}' may not be compatible with current system")
                    for warning in compatibility['warnings']:
                        logger.warning(f"Compatibility warning: {warning}")
            
            # Cache the preset
            self._preset_cache[preset_name] = preset
            
            return preset
            
        except Exception as e:
            logger.error(f"Error loading preset '{preset_name}': {e}")
            raise
    
    def save_preset(self, preset: PresetConfig, preset_name: Optional[str] = None,
                   overwrite: bool = False) -> str:
        """
        Save preset configuration to file.
        
        Args:
            preset: PresetConfig object to save
            preset_name: Optional name override
            overwrite: Whether to overwrite existing preset
            
        Returns:
            Path to saved preset file
            
        Raises:
            FileExistsError: If preset exists and overwrite is False
            PresetValidationError: If preset is invalid
            OSError: If file operations fail
        """
        try:
            name = preset_name or preset.name
            preset_file = self.presets_dir / f"{name}.yaml"
            
            # Check if preset already exists
            if preset_file.exists() and not overwrite:
                raise FileExistsError(f"Preset '{name}' already exists. Use overwrite=True to replace.")
            
            # Validate preset before saving
            self._validate_preset(preset)
            
            # Update metadata
            preset.created_at = datetime.now().isoformat()
            if preset.created_by == "system":
                preset.created_by = "user"
            
            # Save preset
            self.yaml_loader.save_preset(preset, preset_file)
            
            # Update cache
            self._preset_cache[name] = preset
            
            logger.info(f"Saved preset '{name}' to {preset_file}")
            return str(preset_file)
            
        except Exception as e:
            logger.error(f"Error saving preset: {e}")
            raise
    
    def create_custom_preset(self, name: str, description: str,
                           base_preset: str = "theater_medium_balanced",
                           overrides: Optional[Dict[str, Any]] = None) -> PresetConfig:
        """
        Create custom preset based on existing preset with parameter overrides.
        
        Args:
            name: Name for new preset
            description: Description of preset
            base_preset: Base preset to start from
            overrides: Dictionary of parameter overrides
            
        Returns:
            New PresetConfig object
            
        Raises:
            PresetValidationError: If parameters are invalid
        """
        try:
            # Load base preset
            base = self.load_preset(base_preset)
            
            # Create new preset with base parameters
            preset_dict = {
                'name': name,
                'description': description,
                'theater_size': base.theater_size,
                'quality_level': base.quality_level,
                'target_resolution': base.target_resolution,
                'denoise_strength': base.denoise_strength,
                'dialogue_boost': base.dialogue_boost,
                'noise_reduction': base.noise_reduction,
                'reverb_reduction': base.reverb_reduction,
                'preserve_naturalness': base.preserve_naturalness,
                'tile_size': base.tile_size,
                'batch_size': base.batch_size,
                'memory_usage': base.memory_usage,
                'created_by': 'user',
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # Apply overrides
            if overrides:
                for key, value in overrides.items():
                    if key in preset_dict:
                        preset_dict[key] = value
                    else:
                        logger.warning(f"Unknown parameter in overrides: {key}")
            
            # Create and validate preset
            preset = PresetConfig(**preset_dict)
            self._validate_preset(preset)
            
            logger.info(f"Created custom preset '{name}' based on '{base_preset}'")
            return preset
            
        except Exception as e:
            logger.error(f"Error creating custom preset: {e}")
            raise PresetValidationError(f"Failed to create custom preset: {e}")
    
    def apply_parameter_overrides(self, preset: PresetConfig, 
                                overrides: Dict[str, Any]) -> PresetConfig:
        """
        Apply parameter overrides to existing preset.
        
        Args:
            preset: Base preset configuration
            overrides: Dictionary of parameter overrides
            
        Returns:
            New PresetConfig with overrides applied
            
        Raises:
            PresetValidationError: If overrides are invalid
        """
        try:
            # Convert preset to dictionary
            preset_dict = {
                'name': preset.name,
                'description': preset.description,
                'theater_size': preset.theater_size,
                'quality_level': preset.quality_level,
                'target_resolution': preset.target_resolution,
                'denoise_strength': preset.denoise_strength,
                'dialogue_boost': preset.dialogue_boost,
                'noise_reduction': preset.noise_reduction,
                'reverb_reduction': preset.reverb_reduction,
                'preserve_naturalness': preset.preserve_naturalness,
                'tile_size': preset.tile_size,
                'batch_size': preset.batch_size,
                'memory_usage': preset.memory_usage,
                'created_by': preset.created_by,
                'created_at': preset.created_at,
                'version': preset.version
            }
            
            # Validate and apply overrides
            validated_overrides = self._validate_parameter_overrides(overrides)
            preset_dict.update(validated_overrides)
            
            # Create new preset
            new_preset = PresetConfig(**preset_dict)
            
            logger.debug(f"Applied {len(overrides)} parameter overrides to preset '{preset.name}'")
            return new_preset
            
        except Exception as e:
            logger.error(f"Error applying parameter overrides: {e}")
            raise PresetValidationError(f"Failed to apply overrides: {e}")
    
    def list_available_presets(self) -> List[str]:
        """
        List all available presets (file-based and default).
        
        Returns:
            List of preset names
        """
        presets = set()
        
        try:
            # Add file-based presets
            for preset_file in self.presets_dir.glob("*.yaml"):
                presets.add(preset_file.stem)
            
            # Add default presets
            default_presets = get_available_presets()
            presets.update(default_presets.keys())
            
        except Exception as e:
            logger.error(f"Error listing presets: {e}")
        
        return sorted(list(presets))
    
    def get_preset_info(self, preset_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a preset.
        
        Args:
            preset_name: Name of preset
            
        Returns:
            Dictionary containing preset information
        """
        try:
            # Check cache first
            if preset_name in self._preset_metadata_cache:
                return self._preset_metadata_cache[preset_name]
            
            preset = self.load_preset(preset_name)
            
            info = {
                'name': preset.name,
                'description': preset.description,
                'theater_size': preset.theater_size,
                'quality_level': preset.quality_level,
                'target_resolution': preset.target_resolution,
                'created_by': preset.created_by,
                'created_at': preset.created_at,
                'version': preset.version,
                'file_exists': (self.presets_dir / f"{preset_name}.yaml").exists(),
                'is_default': preset_name in get_available_presets()
            }
            
            # Add compatibility information
            if self.system_config:
                compatibility = self._check_preset_compatibility(preset)
                info['compatibility'] = compatibility
            
            # Cache the info
            self._preset_metadata_cache[preset_name] = info
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting preset info for '{preset_name}': {e}")
            return {
                'name': preset_name,
                'error': str(e),
                'file_exists': False,
                'is_default': False
            }
    
    def delete_preset(self, preset_name: str, force: bool = False) -> bool:
        """
        Delete preset file.
        
        Args:
            preset_name: Name of preset to delete
            force: Whether to force deletion of system presets
            
        Returns:
            True if preset was deleted
            
        Raises:
            PermissionError: If trying to delete system preset without force
            OSError: If file operations fail
        """
        try:
            preset_file = self.presets_dir / f"{preset_name}.yaml"
            
            # Check if it's a system preset
            if not force and preset_name in get_available_presets():
                raise PermissionError(
                    f"Cannot delete system preset '{preset_name}' without force=True"
                )
            
            if preset_file.exists():
                # Create backup before deletion
                backup_file = preset_file.with_suffix(".yaml.deleted")
                shutil.copy2(preset_file, backup_file)
                
                # Delete the file
                preset_file.unlink()
                
                # Remove from caches
                self._preset_cache.pop(preset_name, None)
                self._preset_metadata_cache.pop(preset_name, None)
                
                logger.info(f"Deleted preset '{preset_name}' (backup: {backup_file})")
                return True
            else:
                logger.warning(f"Preset file '{preset_name}' not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting preset '{preset_name}': {e}")
            raise
    
    def validate_preset(self, preset: PresetConfig) -> Dict[str, Any]:
        """
        Validate preset configuration.
        
        Args:
            preset: PresetConfig to validate
            
        Returns:
            Dictionary containing validation results
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'compatibility': None
        }
        
        try:
            # Basic validation
            self._validate_preset(preset)
            
            # Compatibility check
            if self.system_config:
                compatibility = self._check_preset_compatibility(preset)
                result['compatibility'] = compatibility
                
                if not compatibility['compatible']:
                    result['warnings'].extend(compatibility['warnings'])
            
        except PresetValidationError as e:
            result['valid'] = False
            result['errors'].append(str(e))
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {e}")
        
        return result
    
    def export_preset(self, preset_name: str, export_path: Union[str, Path]) -> str:
        """
        Export preset to specified location.
        
        Args:
            preset_name: Name of preset to export
            export_path: Path to export preset
            
        Returns:
            Path to exported preset file
            
        Raises:
            FileNotFoundError: If preset not found
            OSError: If export fails
        """
        try:
            preset = self.load_preset(preset_name)
            export_file = Path(export_path)
            
            # Ensure export directory exists
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save preset to export location
            self.yaml_loader.save_preset(preset, export_file)
            
            logger.info(f"Exported preset '{preset_name}' to {export_file}")
            return str(export_file)
            
        except Exception as e:
            logger.error(f"Error exporting preset '{preset_name}': {e}")
            raise
    
    def import_preset(self, import_path: Union[str, Path], 
                     preset_name: Optional[str] = None,
                     overwrite: bool = False) -> str:
        """
        Import preset from file.
        
        Args:
            import_path: Path to preset file to import
            preset_name: Optional name for imported preset
            overwrite: Whether to overwrite existing preset
            
        Returns:
            Name of imported preset
            
        Raises:
            FileNotFoundError: If import file not found
            PresetValidationError: If preset is invalid
            FileExistsError: If preset exists and overwrite is False
        """
        try:
            import_file = Path(import_path)
            if not import_file.exists():
                raise FileNotFoundError(f"Import file not found: {import_path}")
            
            # Load preset from import file
            preset = self.yaml_loader.load_preset(import_file)
            
            # Use provided name or preset's name
            name = preset_name or preset.name
            preset.name = name
            
            # Save to presets directory
            saved_path = self.save_preset(preset, name, overwrite=overwrite)
            
            logger.info(f"Imported preset '{name}' from {import_path}")
            return name
            
        except Exception as e:
            logger.error(f"Error importing preset from '{import_path}': {e}")
            raise
    
    def clear_cache(self) -> None:
        """Clear preset cache."""
        self._preset_cache.clear()
        self._preset_metadata_cache.clear()
        logger.debug("Preset cache cleared")
    
    def _validate_preset(self, preset: PresetConfig) -> None:
        """
        Validate preset configuration.
        
        Args:
            preset: PresetConfig to validate
            
        Raises:
            PresetValidationError: If preset is invalid
        """
        try:
            # Basic dataclass validation is done in __post_init__
            # Additional validation can be added here
            
            # Check target resolution format
            if preset.target_resolution:
                if not self._validate_resolution_format(preset.target_resolution):
                    raise PresetValidationError(
                        f"Invalid target resolution format: {preset.target_resolution}"
                    )
            
            # Check parameter ranges
            if not 0.0 <= preset.denoise_strength <= 1.0:
                raise PresetValidationError("denoise_strength must be between 0.0 and 1.0")
            
            if not 0.0 <= preset.dialogue_boost <= 20.0:
                raise PresetValidationError("dialogue_boost must be between 0.0 and 20.0")
            
            if not 64 <= preset.tile_size <= 2048:
                raise PresetValidationError("tile_size must be between 64 and 2048")
            
            if not 1 <= preset.batch_size <= 16:
                raise PresetValidationError("batch_size must be between 1 and 16")
            
        except Exception as e:
            if isinstance(e, PresetValidationError):
                raise
            else:
                raise PresetValidationError(f"Preset validation failed: {e}")
    
    def _validate_parameter_overrides(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameter overrides.
        
        Args:
            overrides: Dictionary of parameter overrides
            
        Returns:
            Validated overrides dictionary
            
        Raises:
            PresetValidationError: If overrides are invalid
        """
        validated = {}
        
        # Define valid parameters and their validation functions
        validators = {
            'theater_size': lambda x: x in {'small', 'medium', 'large'},
            'quality_level': lambda x: x in {'fast', 'balanced', 'high'},
            'denoise_strength': lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 1.0,
            'dialogue_boost': lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 20.0,
            'noise_reduction': lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 1.0,
            'reverb_reduction': lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 1.0,
            'preserve_naturalness': lambda x: isinstance(x, bool),
            'tile_size': lambda x: isinstance(x, int) and 64 <= x <= 2048,
            'batch_size': lambda x: isinstance(x, int) and 1 <= x <= 16,
            'memory_usage': lambda x: isinstance(x, (int, float)) and 0.1 <= x <= 0.9,
            'target_resolution': lambda x: x is None or self._validate_resolution_format(x)
        }
        
        for key, value in overrides.items():
            if key not in validators:
                logger.warning(f"Unknown parameter in overrides: {key}")
                continue
            
            if not validators[key](value):
                raise PresetValidationError(f"Invalid value for parameter '{key}': {value}")
            
            validated[key] = value
        
        return validated
    
    def _validate_resolution_format(self, resolution: str) -> bool:
        """Validate resolution format (e.g., '3840x2160')."""
        import re
        pattern = r'^\d{3,5}x\d{3,5}$'
        return bool(re.match(pattern, resolution))
    
    def _check_preset_compatibility(self, preset: PresetConfig) -> Dict[str, Any]:
        """
        Check preset compatibility with current system.
        
        Args:
            preset: PresetConfig to check
            
        Returns:
            Dictionary containing compatibility information
        """
        if not self.system_config:
            return {'compatible': True, 'warnings': [], 'recommendations': []}
        
        return validate_preset_compatibility(preset.__dict__)