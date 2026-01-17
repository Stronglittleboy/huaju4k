"""
Enhanced YAML configuration loader with validation and error handling.

This module provides robust YAML configuration loading with comprehensive
validation, error handling, and parameter checking.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
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

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    
    def __init__(self, message: str, errors: List[str] = None, warnings: List[str] = None):
        super().__init__(message)
        self.errors = errors or []
        self.warnings = warnings or []


class YAMLConfigLoader:
    """
    Enhanced YAML configuration loader with validation and error handling.
    
    Provides robust loading of YAML configuration files with comprehensive
    validation, parameter checking, and fallback to defaults.
    """
    
    def __init__(self, strict_validation: bool = True, 
                 allow_unknown_keys: bool = False):
        """
        Initialize YAML configuration loader.
        
        Args:
            strict_validation: Whether to enforce strict validation
            allow_unknown_keys: Whether to allow unknown configuration keys
        """
        self.strict_validation = strict_validation
        self.allow_unknown_keys = allow_unknown_keys
        
        if not HAS_YAML:
            logger.warning("YAML support not available - install PyYAML for full functionality")
    
    def load_config(self, config_path: Union[str, Path], 
                   fallback_to_defaults: bool = True) -> HuaJu4KConfig:
        """
        Load configuration from YAML file with validation.
        
        Args:
            config_path: Path to configuration file
            fallback_to_defaults: Whether to use defaults if file not found
            
        Returns:
            HuaJu4KConfig object
            
        Raises:
            ConfigValidationError: If configuration is invalid
            FileNotFoundError: If file not found and fallback disabled
            OSError: If file operations fail
        """
        config_file = Path(config_path)
        
        try:
            # Check if file exists
            if not config_file.exists():
                if fallback_to_defaults:
                    logger.info(f"Configuration file {config_path} not found, using defaults")
                    return self._create_default_config()
                else:
                    raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            # Load YAML content
            config_data = self._load_yaml_file(config_file)
            
            # Validate configuration structure
            validation_result = self._validate_config_structure(config_data)
            
            if not validation_result['valid']:
                if self.strict_validation:
                    raise ConfigValidationError(
                        f"Configuration validation failed: {validation_result['errors']}",
                        errors=validation_result['errors'],
                        warnings=validation_result.get('warnings', [])
                    )
                else:
                    logger.warning(f"Configuration validation warnings: {validation_result['errors']}")
            
            # Log warnings
            for warning in validation_result.get('warnings', []):
                logger.warning(f"Configuration warning: {warning}")
            
            # Create configuration object
            config = HuaJu4KConfig.from_dict(config_data)
            
            # Additional validation
            self._validate_config_values(config)
            
            logger.info(f"Successfully loaded configuration from {config_path}")
            return config
            
        except yaml.YAMLError as e:
            error_msg = f"YAML parsing error in {config_path}: {e}"
            logger.error(error_msg)
            raise ConfigValidationError(error_msg, errors=[str(e)])
        
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            raise
    
    def load_preset(self, preset_path: Union[str, Path]) -> PresetConfig:
        """
        Load preset configuration from YAML file.
        
        Args:
            preset_path: Path to preset file
            
        Returns:
            PresetConfig object
            
        Raises:
            ConfigValidationError: If preset is invalid
            FileNotFoundError: If preset file not found
        """
        preset_file = Path(preset_path)
        
        try:
            if not preset_file.exists():
                raise FileNotFoundError(f"Preset file not found: {preset_path}")
            
            # Load YAML content
            preset_data = self._load_yaml_file(preset_file)
            
            # Validate preset structure
            validation_result = self._validate_preset_structure(preset_data)
            
            if not validation_result['valid']:
                raise ConfigValidationError(
                    f"Preset validation failed: {validation_result['errors']}",
                    errors=validation_result['errors'],
                    warnings=validation_result.get('warnings', [])
                )
            
            # Create preset object
            preset = PresetConfig(**preset_data)
            
            logger.info(f"Successfully loaded preset from {preset_path}")
            return preset
            
        except yaml.YAMLError as e:
            error_msg = f"YAML parsing error in {preset_path}: {e}"
            logger.error(error_msg)
            raise ConfigValidationError(error_msg, errors=[str(e)])
        
        except Exception as e:
            logger.error(f"Error loading preset from {preset_path}: {e}")
            raise
    
    def save_config(self, config: HuaJu4KConfig, config_path: Union[str, Path],
                   create_backup: bool = True) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            config_path: Path to save configuration
            create_backup: Whether to create backup of existing file
            
        Raises:
            OSError: If file operations fail
        """
        if not HAS_YAML:
            raise ImportError("YAML support not available - install PyYAML")
        
        config_file = Path(config_path)
        
        try:
            # Create backup if requested and file exists
            if create_backup and config_file.exists():
                backup_path = config_file.with_suffix(f"{config_file.suffix}.backup")
                config_file.rename(backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Ensure directory exists
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert config to dictionary
            config_dict = config.to_dict()
            
            # Add metadata
            config_dict['_metadata'] = {
                'version': '1.0',
                'created_by': 'huaju4k',
                'schema_version': '1.0'
            }
            
            # Save to file with proper formatting
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config_dict, 
                    f, 
                    default_flow_style=False,
                    indent=2,
                    sort_keys=False,
                    allow_unicode=True
                )
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
            raise
    
    def save_preset(self, preset: PresetConfig, preset_path: Union[str, Path]) -> None:
        """
        Save preset configuration to YAML file.
        
        Args:
            preset: Preset to save
            preset_path: Path to save preset
            
        Raises:
            OSError: If file operations fail
        """
        if not HAS_YAML:
            raise ImportError("YAML support not available - install PyYAML")
        
        preset_file = Path(preset_path)
        
        try:
            # Ensure directory exists
            preset_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save preset
            preset.save_to_file(str(preset_file))
            
            logger.info(f"Preset saved to {preset_path}")
            
        except Exception as e:
            logger.error(f"Error saving preset to {preset_path}: {e}")
            raise
    
    def validate_config_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate configuration file without loading it.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing validation results
        """
        config_file = Path(config_path)
        
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'file_exists': config_file.exists(),
            'file_readable': False,
            'yaml_valid': False,
            'structure_valid': False
        }
        
        try:
            # Check file existence and readability
            if not config_file.exists():
                result['errors'].append(f"Configuration file not found: {config_path}")
                return result
            
            if not os.access(config_file, os.R_OK):
                result['errors'].append(f"Configuration file not readable: {config_path}")
                return result
            
            result['file_readable'] = True
            
            # Check YAML validity
            try:
                config_data = self._load_yaml_file(config_file)
                result['yaml_valid'] = True
            except yaml.YAMLError as e:
                result['errors'].append(f"YAML parsing error: {e}")
                return result
            
            # Check structure validity
            validation_result = self._validate_config_structure(config_data)
            result['structure_valid'] = validation_result['valid']
            result['errors'].extend(validation_result['errors'])
            result['warnings'].extend(validation_result.get('warnings', []))
            
            result['valid'] = result['structure_valid']
            
        except Exception as e:
            result['errors'].append(f"Validation error: {e}")
        
        return result
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse YAML file."""
        if not HAS_YAML:
            raise ImportError("YAML support not available")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for common YAML issues
        self._check_yaml_syntax(content, str(file_path))
        
        # Parse YAML
        data = yaml.safe_load(content)
        
        if data is None:
            return {}
        
        if not isinstance(data, dict):
            raise ValueError("Configuration file must contain a YAML dictionary")
        
        return data
    
    def _check_yaml_syntax(self, content: str, file_path: str) -> None:
        """Check for common YAML syntax issues."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for tabs (should use spaces)
            if '\t' in line:
                logger.warning(f"{file_path}:{i}: Using tabs instead of spaces for indentation")
            
            # Check for trailing whitespace
            if line.rstrip() != line:
                logger.warning(f"{file_path}:{i}: Trailing whitespace detected")
            
            # Check for very long lines
            if len(line) > 120:
                logger.warning(f"{file_path}:{i}: Very long line ({len(line)} characters)")
    
    def _validate_config_structure(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration structure."""
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check required sections
            required_sections = ['video', 'audio', 'performance']
            for section in required_sections:
                if section not in config_data:
                    result['warnings'].append(f"Missing section '{section}', will use defaults")
            
            # Validate video section
            if 'video' in config_data:
                video_result = self._validate_video_config(config_data['video'])
                result['errors'].extend(video_result['errors'])
                result['warnings'].extend(video_result['warnings'])
            
            # Validate audio section
            if 'audio' in config_data:
                audio_result = self._validate_audio_config(config_data['audio'])
                result['errors'].extend(audio_result['errors'])
                result['warnings'].extend(audio_result['warnings'])
            
            # Validate performance section
            if 'performance' in config_data:
                perf_result = self._validate_performance_config(config_data['performance'])
                result['errors'].extend(perf_result['errors'])
                result['warnings'].extend(perf_result['warnings'])
            
            # Check for unknown keys if not allowed
            if not self.allow_unknown_keys:
                known_keys = {
                    'video', 'audio', 'performance', 'logging', 'paths', 'ui', '_metadata',
                    'log_level', 'log_file', 'temp_dir', 'models_dir', 'presets_dir',
                    'progress_update_interval', 'auto_cleanup', 'save_processing_logs'
                }
                
                unknown_keys = set(config_data.keys()) - known_keys
                if unknown_keys:
                    result['warnings'].extend([
                        f"Unknown configuration key: {key}" for key in unknown_keys
                    ])
            
            result['valid'] = len(result['errors']) == 0
            
        except Exception as e:
            result['errors'].append(f"Structure validation error: {e}")
            result['valid'] = False
        
        return result
    
    def _validate_video_config(self, video_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate video configuration section."""
        result = {'errors': [], 'warnings': []}
        
        # Check AI model
        if 'ai_model' in video_config:
            valid_models = {'real_esrgan', 'esrgan', 'waifu2x'}
            if video_config['ai_model'] not in valid_models:
                result['warnings'].append(
                    f"Unknown AI model '{video_config['ai_model']}', "
                    f"valid options: {valid_models}"
                )
        
        # Check quality presets
        if 'quality_presets' in video_config:
            presets = video_config['quality_presets']
            if not isinstance(presets, dict):
                result['errors'].append("quality_presets must be a dictionary")
            else:
                for preset_name, preset_config in presets.items():
                    if not isinstance(preset_config, dict):
                        result['errors'].append(f"Quality preset '{preset_name}' must be a dictionary")
                    else:
                        # Validate preset parameters
                        if 'tile_size' in preset_config:
                            tile_size = preset_config['tile_size']
                            if not isinstance(tile_size, int) or tile_size < 64 or tile_size > 2048:
                                result['errors'].append(
                                    f"Invalid tile_size in preset '{preset_name}': must be integer 64-2048"
                                )
        
        return result
    
    def _validate_audio_config(self, audio_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate audio configuration section."""
        result = {'errors': [], 'warnings': []}
        
        # Check sample rate
        if 'sample_rate' in audio_config:
            sample_rate = audio_config['sample_rate']
            valid_rates = {22050, 44100, 48000, 96000}
            if sample_rate not in valid_rates:
                result['warnings'].append(
                    f"Unusual sample rate {sample_rate}, "
                    f"common rates: {valid_rates}"
                )
        
        # Check theater presets
        if 'theater_presets' in audio_config:
            presets = audio_config['theater_presets']
            if not isinstance(presets, dict):
                result['errors'].append("theater_presets must be a dictionary")
            else:
                for preset_name, preset_config in presets.items():
                    if not isinstance(preset_config, dict):
                        result['errors'].append(f"Theater preset '{preset_name}' must be a dictionary")
                    else:
                        # Validate preset parameters
                        for param in ['dialogue_boost', 'noise_reduction', 'reverb_reduction']:
                            if param in preset_config:
                                value = preset_config[param]
                                if not isinstance(value, (int, float)):
                                    result['errors'].append(
                                        f"Parameter '{param}' in preset '{preset_name}' must be numeric"
                                    )
        
        return result
    
    def _validate_performance_config(self, perf_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate performance configuration section."""
        result = {'errors': [], 'warnings': []}
        
        # Check memory usage
        if 'max_memory_usage' in perf_config:
            memory_usage = perf_config['max_memory_usage']
            if not isinstance(memory_usage, (int, float)) or not 0.1 <= memory_usage <= 0.9:
                result['errors'].append("max_memory_usage must be a number between 0.1 and 0.9")
        
        # Check worker count
        if 'num_workers' in perf_config:
            num_workers = perf_config['num_workers']
            if not isinstance(num_workers, int) or num_workers < 1 or num_workers > 32:
                result['errors'].append("num_workers must be an integer between 1 and 32")
        
        # Check GPU settings
        if 'use_gpu' in perf_config and perf_config['use_gpu']:
            if 'gpu_id' in perf_config:
                gpu_id = perf_config['gpu_id']
                if not isinstance(gpu_id, int) or gpu_id < 0:
                    result['errors'].append("gpu_id must be a non-negative integer")
        
        return result
    
    def _validate_preset_structure(self, preset_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate preset structure."""
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check required fields
            required_fields = ['name', 'description', 'theater_size', 'quality_level']
            for field in required_fields:
                if field not in preset_data:
                    result['errors'].append(f"Missing required field: {field}")
            
            # Validate theater size
            if 'theater_size' in preset_data:
                valid_sizes = {'small', 'medium', 'large'}
                if preset_data['theater_size'] not in valid_sizes:
                    result['errors'].append(
                        f"Invalid theater_size '{preset_data['theater_size']}', "
                        f"valid options: {valid_sizes}"
                    )
            
            # Validate quality level
            if 'quality_level' in preset_data:
                valid_levels = {'fast', 'balanced', 'high'}
                if preset_data['quality_level'] not in valid_levels:
                    result['errors'].append(
                        f"Invalid quality_level '{preset_data['quality_level']}', "
                        f"valid options: {valid_levels}"
                    )
            
            # Validate numeric ranges
            numeric_validations = {
                'denoise_strength': (0.0, 1.0),
                'dialogue_boost': (0.0, 20.0),
                'noise_reduction': (0.0, 1.0),
                'reverb_reduction': (0.0, 1.0),
                'memory_usage': (0.1, 0.9)
            }
            
            for field, (min_val, max_val) in numeric_validations.items():
                if field in preset_data:
                    value = preset_data[field]
                    if not isinstance(value, (int, float)) or not min_val <= value <= max_val:
                        result['errors'].append(
                            f"Field '{field}' must be a number between {min_val} and {max_val}"
                        )
            
            result['valid'] = len(result['errors']) == 0
            
        except Exception as e:
            result['errors'].append(f"Preset validation error: {e}")
            result['valid'] = False
        
        return result
    
    def _validate_config_values(self, config: HuaJu4KConfig) -> None:
        """Validate configuration values after object creation."""
        # Additional validation can be added here
        # This is called after the config object is created
        pass
    
    def _create_default_config(self) -> HuaJu4KConfig:
        """Create default configuration."""
        default_dict = get_default_config()
        return HuaJu4KConfig.from_dict(default_dict)