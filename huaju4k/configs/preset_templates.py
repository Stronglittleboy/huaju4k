"""
Preset templates and theater-specific configurations for huaju4k.

This module provides comprehensive preset templates for different theater
scenarios and processing requirements.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class TheaterSize(Enum):
    """Theater size categories."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class QualityLevel(Enum):
    """Processing quality levels."""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"


@dataclass
class TheaterCharacteristics:
    """Characteristics of different theater sizes."""
    
    size: TheaterSize
    typical_capacity: int
    stage_distance_meters: float
    reverb_time_seconds: float
    ambient_noise_level: float
    recommended_dialogue_boost: float
    recommended_noise_reduction: float
    
    @property
    def description(self) -> str:
        """Get description of theater characteristics."""
        return (
            f"{self.size.value.title()} theater "
            f"({self.typical_capacity} seats, "
            f"{self.stage_distance_meters}m stage distance)"
        )


# Theater size characteristics
THEATER_CHARACTERISTICS = {
    TheaterSize.SMALL: TheaterCharacteristics(
        size=TheaterSize.SMALL,
        typical_capacity=150,
        stage_distance_meters=8.0,
        reverb_time_seconds=1.2,
        ambient_noise_level=0.3,
        recommended_dialogue_boost=6.0,
        recommended_noise_reduction=0.7
    ),
    TheaterSize.MEDIUM: TheaterCharacteristics(
        size=TheaterSize.MEDIUM,
        typical_capacity=400,
        stage_distance_meters=15.0,
        reverb_time_seconds=1.8,
        ambient_noise_level=0.4,
        recommended_dialogue_boost=4.0,
        recommended_noise_reduction=0.5
    ),
    TheaterSize.LARGE: TheaterCharacteristics(
        size=TheaterSize.LARGE,
        typical_capacity=800,
        stage_distance_meters=25.0,
        reverb_time_seconds=2.5,
        ambient_noise_level=0.5,
        recommended_dialogue_boost=2.0,
        recommended_noise_reduction=0.3
    )
}


def get_theater_characteristics(theater_size: str) -> TheaterCharacteristics:
    """
    Get characteristics for theater size.
    
    Args:
        theater_size: Theater size string
        
    Returns:
        TheaterCharacteristics object
        
    Raises:
        ValueError: If theater size is invalid
    """
    try:
        size_enum = TheaterSize(theater_size.lower())
        return THEATER_CHARACTERISTICS[size_enum]
    except ValueError:
        valid_sizes = [size.value for size in TheaterSize]
        raise ValueError(f"Invalid theater size '{theater_size}'. Valid sizes: {valid_sizes}")


def generate_preset_template(theater_size: str, quality_level: str, 
                           custom_name: str = None) -> Dict[str, Any]:
    """
    Generate preset template based on theater size and quality level.
    
    Args:
        theater_size: Theater size (small, medium, large)
        quality_level: Quality level (fast, balanced, high)
        custom_name: Optional custom name for preset
        
    Returns:
        Dictionary containing preset configuration
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Validate inputs
    try:
        size_enum = TheaterSize(theater_size.lower())
        quality_enum = QualityLevel(quality_level.lower())
    except ValueError as e:
        raise ValueError(f"Invalid parameter: {e}")
    
    # Get theater characteristics
    characteristics = THEATER_CHARACTERISTICS[size_enum]
    
    # Base preset name
    if custom_name:
        name = custom_name
    else:
        name = f"Theater {size_enum.value.title()} - {quality_enum.value.title()}"
    
    # Quality-specific parameters
    quality_params = _get_quality_parameters(quality_enum)
    
    # Theater-specific audio parameters
    audio_params = _get_theater_audio_parameters(characteristics)
    
    # Generate preset
    preset = {
        'name': name,
        'description': f"Optimized for {characteristics.description} with {quality_enum.value} processing",
        'theater_size': size_enum.value,
        'quality_level': quality_enum.value,
        'target_resolution': '3840x2160',
        
        # Video parameters
        'denoise_strength': quality_params['denoise_strength'],
        'tile_size': quality_params['tile_size'],
        'batch_size': quality_params['batch_size'],
        'memory_usage': quality_params['memory_usage'],
        
        # Audio parameters
        'dialogue_boost': audio_params['dialogue_boost'],
        'noise_reduction': audio_params['noise_reduction'],
        'reverb_reduction': audio_params['reverb_reduction'],
        'preserve_naturalness': True,
        
        # Metadata
        'created_by': 'system',
        'version': '1.0'
    }
    
    return preset


def _get_quality_parameters(quality_level: QualityLevel) -> Dict[str, Any]:
    """Get quality-specific processing parameters."""
    
    quality_configs = {
        QualityLevel.FAST: {
            'denoise_strength': 0.5,
            'tile_size': 512,
            'batch_size': 4,
            'memory_usage': 0.6,
            'processing_priority': 'speed'
        },
        QualityLevel.BALANCED: {
            'denoise_strength': 0.7,
            'tile_size': 768,
            'batch_size': 2,
            'memory_usage': 0.7,
            'processing_priority': 'balanced'
        },
        QualityLevel.HIGH: {
            'denoise_strength': 0.9,
            'tile_size': 1024,
            'batch_size': 1,
            'memory_usage': 0.8,
            'processing_priority': 'quality'
        }
    }
    
    return quality_configs[quality_level]


def _get_theater_audio_parameters(characteristics: TheaterCharacteristics) -> Dict[str, Any]:
    """Get theater-specific audio parameters."""
    
    # Base parameters on theater characteristics
    reverb_reduction = min(0.9, characteristics.reverb_time_seconds / 3.0)
    spatial_enhancement = 0.3 + (characteristics.stage_distance_meters / 50.0)
    
    return {
        'dialogue_boost': characteristics.recommended_dialogue_boost,
        'noise_reduction': characteristics.recommended_noise_reduction,
        'reverb_reduction': reverb_reduction,
        'spatial_enhancement': min(0.8, spatial_enhancement),
        'dynamic_range_compression': 0.2 + (characteristics.ambient_noise_level * 0.3)
    }


def get_recommended_presets() -> List[Dict[str, Any]]:
    """
    Get list of recommended preset configurations.
    
    Returns:
        List of preset dictionaries
    """
    recommended = []
    
    # Generate recommended combinations
    combinations = [
        (TheaterSize.SMALL, QualityLevel.FAST),
        (TheaterSize.SMALL, QualityLevel.BALANCED),
        (TheaterSize.MEDIUM, QualityLevel.BALANCED),
        (TheaterSize.MEDIUM, QualityLevel.HIGH),
        (TheaterSize.LARGE, QualityLevel.BALANCED),
        (TheaterSize.LARGE, QualityLevel.HIGH)
    ]
    
    for theater_size, quality_level in combinations:
        preset = generate_preset_template(theater_size.value, quality_level.value)
        recommended.append(preset)
    
    return recommended


def validate_preset_parameters(preset_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate preset parameters against theater-specific requirements.
    
    Args:
        preset_dict: Preset configuration dictionary
        
    Returns:
        Dictionary containing validation results and recommendations
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'recommendations': []
    }
    
    try:
        # Check required fields
        required_fields = ['theater_size', 'quality_level', 'dialogue_boost', 'noise_reduction']
        for field in required_fields:
            if field not in preset_dict:
                result['errors'].append(f"Missing required field: {field}")
        
        if result['errors']:
            result['valid'] = False
            return result
        
        # Get theater characteristics for validation
        try:
            characteristics = get_theater_characteristics(preset_dict['theater_size'])
        except ValueError as e:
            result['errors'].append(str(e))
            result['valid'] = False
            return result
        
        # Validate audio parameters against theater characteristics
        dialogue_boost = preset_dict.get('dialogue_boost', 0)
        recommended_boost = characteristics.recommended_dialogue_boost
        
        if abs(dialogue_boost - recommended_boost) > 3.0:
            result['warnings'].append(
                f"Dialogue boost ({dialogue_boost}) differs significantly from "
                f"recommended value ({recommended_boost}) for {characteristics.size.value} theater"
            )
            result['recommendations'].append(
                f"Consider using dialogue boost around {recommended_boost} for optimal results"
            )
        
        # Validate noise reduction
        noise_reduction = preset_dict.get('noise_reduction', 0)
        recommended_nr = characteristics.recommended_noise_reduction
        
        if abs(noise_reduction - recommended_nr) > 0.3:
            result['warnings'].append(
                f"Noise reduction ({noise_reduction}) may not be optimal for "
                f"{characteristics.size.value} theater (recommended: {recommended_nr})"
            )
        
        # Validate quality level consistency
        quality_level = preset_dict.get('quality_level', 'balanced')
        tile_size = preset_dict.get('tile_size', 768)
        
        quality_params = _get_quality_parameters(QualityLevel(quality_level))
        expected_tile_size = quality_params['tile_size']
        
        if abs(tile_size - expected_tile_size) > 256:
            result['warnings'].append(
                f"Tile size ({tile_size}) inconsistent with quality level ({quality_level}). "
                f"Expected around {expected_tile_size}"
            )
        
        # Check memory usage appropriateness
        memory_usage = preset_dict.get('memory_usage', 0.7)
        if quality_level == 'high' and memory_usage < 0.7:
            result['recommendations'].append(
                "High quality processing typically requires memory usage >= 0.7"
            )
        elif quality_level == 'fast' and memory_usage > 0.7:
            result['recommendations'].append(
                "Fast processing can use lower memory usage (0.5-0.6) for better performance"
            )
        
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Validation error: {e}")
    
    return result


def get_preset_compatibility_info(preset_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get compatibility information for preset.
    
    Args:
        preset_dict: Preset configuration dictionary
        
    Returns:
        Dictionary containing compatibility information
    """
    info = {
        'theater_info': None,
        'quality_info': None,
        'memory_requirements': None,
        'processing_time_estimate': None
    }
    
    try:
        # Theater information
        theater_size = preset_dict.get('theater_size', 'medium')
        characteristics = get_theater_characteristics(theater_size)
        info['theater_info'] = {
            'size': characteristics.size.value,
            'description': characteristics.description,
            'typical_capacity': characteristics.typical_capacity,
            'stage_distance': characteristics.stage_distance_meters,
            'reverb_time': characteristics.reverb_time_seconds
        }
        
        # Quality information
        quality_level = preset_dict.get('quality_level', 'balanced')
        quality_params = _get_quality_parameters(QualityLevel(quality_level))
        info['quality_info'] = {
            'level': quality_level,
            'tile_size': quality_params['tile_size'],
            'batch_size': quality_params['batch_size'],
            'priority': quality_params['processing_priority']
        }
        
        # Memory requirements estimation
        tile_size = preset_dict.get('tile_size', 768)
        batch_size = preset_dict.get('batch_size', 2)
        memory_usage = preset_dict.get('memory_usage', 0.7)
        
        # Rough memory calculation (in MB)
        estimated_memory = (tile_size * tile_size * 3 * batch_size * 4) // (1024 * 1024)
        info['memory_requirements'] = {
            'estimated_mb': estimated_memory,
            'memory_usage_factor': memory_usage,
            'recommended_system_memory_mb': int(estimated_memory / memory_usage)
        }
        
        # Processing time estimation (relative)
        time_factors = {
            'fast': 1.0,
            'balanced': 2.0,
            'high': 4.0
        }
        
        base_factor = time_factors.get(quality_level, 2.0)
        theater_factor = {
            'small': 0.8,
            'medium': 1.0,
            'large': 1.2
        }.get(theater_size, 1.0)
        
        info['processing_time_estimate'] = {
            'relative_factor': base_factor * theater_factor,
            'quality_impact': base_factor,
            'theater_impact': theater_factor,
            'description': f"{quality_level.title()} quality for {theater_size} theater"
        }
        
    except Exception as e:
        info['error'] = str(e)
    
    return info


def create_custom_preset_wizard(requirements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create custom preset based on user requirements.
    
    Args:
        requirements: Dictionary containing user requirements
        
    Returns:
        Generated preset configuration
    """
    # Default values
    theater_size = requirements.get('theater_size', 'medium')
    quality_level = requirements.get('quality_level', 'balanced')
    
    # Start with template
    preset = generate_preset_template(theater_size, quality_level)
    
    # Apply custom requirements
    if 'name' in requirements:
        preset['name'] = requirements['name']
    
    if 'description' in requirements:
        preset['description'] = requirements['description']
    
    # Audio customizations
    if 'dialogue_clarity' in requirements:
        clarity_level = requirements['dialogue_clarity']  # 'low', 'medium', 'high'
        if clarity_level == 'high':
            preset['dialogue_boost'] *= 1.3
        elif clarity_level == 'low':
            preset['dialogue_boost'] *= 0.7
    
    if 'background_noise' in requirements:
        noise_level = requirements['background_noise']  # 'low', 'medium', 'high'
        if noise_level == 'high':
            preset['noise_reduction'] = min(0.9, preset['noise_reduction'] * 1.4)
        elif noise_level == 'low':
            preset['noise_reduction'] *= 0.6
    
    # Performance customizations
    if 'processing_speed_priority' in requirements:
        if requirements['processing_speed_priority']:
            preset['tile_size'] = max(256, preset['tile_size'] // 2)
            preset['batch_size'] = min(8, preset['batch_size'] * 2)
            preset['denoise_strength'] *= 0.8
    
    if 'memory_constraint' in requirements:
        memory_limit = requirements['memory_constraint']  # 'low', 'medium', 'high'
        if memory_limit == 'low':
            preset['tile_size'] = min(512, preset['tile_size'])
            preset['batch_size'] = 1
            preset['memory_usage'] = 0.5
        elif memory_limit == 'high':
            preset['memory_usage'] = 0.8
    
    # Ensure values are within valid ranges
    preset['dialogue_boost'] = max(0.0, min(20.0, preset['dialogue_boost']))
    preset['noise_reduction'] = max(0.0, min(1.0, preset['noise_reduction']))
    preset['denoise_strength'] = max(0.0, min(1.0, preset['denoise_strength']))
    preset['tile_size'] = max(64, min(2048, preset['tile_size']))
    preset['batch_size'] = max(1, min(16, preset['batch_size']))
    preset['memory_usage'] = max(0.1, min(0.9, preset['memory_usage']))
    
    # Update metadata
    preset['created_by'] = 'wizard'
    
    return preset