"""
Default configuration and presets for huaju4k video enhancement.

This module provides default configurations and theater presets
that can be used as starting points for video processing.
"""

from typing import Dict, Any

# Default main configuration
DEFAULT_CONFIG = {
    'video': {
        'ai_model': 'real_esrgan',
        'model_path': './models/RealESRGAN_x4plus.pth',
        'quality_presets': {
            'fast': {
                'tile_size': 512,
                'batch_size': 4,
                'denoise_strength': 0.5,
                'overlap_pixels': 16
            },
            'balanced': {
                'tile_size': 768,
                'batch_size': 2,
                'denoise_strength': 0.7,
                'overlap_pixels': 32
            },
            'high': {
                'tile_size': 1024,
                'batch_size': 1,
                'denoise_strength': 0.9,
                'overlap_pixels': 64
            }
        },
        'output': {
            'format': 'mp4',
            'codec': 'h264',
            'crf': 18,
            'preset': 'slow'
        }
    },
    'audio': {
        'theater_presets': {
            'small': {
                'reverb_reduction': 0.8,
                'dialogue_boost': 6.0,
                'noise_reduction': 0.7,
                'spatial_enhancement': 0.3,
                'dynamic_range_compression': 0.2
            },
            'medium': {
                'reverb_reduction': 0.6,
                'dialogue_boost': 4.0,
                'noise_reduction': 0.5,
                'spatial_enhancement': 0.5,
                'dynamic_range_compression': 0.3
            },
            'large': {
                'reverb_reduction': 0.4,
                'dialogue_boost': 2.0,
                'noise_reduction': 0.3,
                'spatial_enhancement': 0.7,
                'dynamic_range_compression': 0.4
            }
        },
        'sample_rate': 48000,
        'bitrate': '192k',
        'channels': 2
    },
    'performance': {
        'use_gpu': True,
        'gpu_id': 0,
        'max_memory_usage': 0.7,
        'tile_overlap': 32,
        'num_workers': 2,
        'prefetch_factor': 2,
        'checkpoint_interval': 60
    },
    'logging': {
        'level': 'INFO',
        'file': None,
        'console': True,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    },
    'paths': {
        'temp_dir': None,  # Use system temp
        'models_dir': './models',
        'presets_dir': './presets',
        'cache_dir': './cache'
    },
    'ui': {
        'progress_update_interval': 0.5,
        'auto_cleanup': True,
        'save_processing_logs': True,
        'show_advanced_options': False
    }
}

# Default theater presets with complete configurations
DEFAULT_PRESETS = {
    'theater_small_fast': {
        'name': 'Small Theater - Fast',
        'description': 'Optimized for small theater venues with fast processing',
        'theater_size': 'small',
        'quality_level': 'fast',
        'target_resolution': '3840x2160',
        'denoise_strength': 0.5,
        'dialogue_boost': 6.0,
        'noise_reduction': 0.7,
        'reverb_reduction': 0.8,
        'preserve_naturalness': True,
        'tile_size': 512,
        'batch_size': 4,
        'memory_usage': 0.6,
        'created_by': 'system',
        'version': '1.0'
    },
    'theater_medium_balanced': {
        'name': 'Medium Theater - Balanced',
        'description': 'Balanced quality and speed for medium theater venues',
        'theater_size': 'medium',
        'quality_level': 'balanced',
        'target_resolution': '3840x2160',
        'denoise_strength': 0.7,
        'dialogue_boost': 4.0,
        'noise_reduction': 0.5,
        'reverb_reduction': 0.6,
        'preserve_naturalness': True,
        'tile_size': 768,
        'batch_size': 2,
        'memory_usage': 0.7,
        'created_by': 'system',
        'version': '1.0'
    },
    'theater_large_high': {
        'name': 'Large Theater - High Quality',
        'description': 'Maximum quality for large theater venues',
        'theater_size': 'large',
        'quality_level': 'high',
        'target_resolution': '3840x2160',
        'denoise_strength': 0.9,
        'dialogue_boost': 2.0,
        'noise_reduction': 0.3,
        'reverb_reduction': 0.4,
        'preserve_naturalness': True,
        'tile_size': 1024,
        'batch_size': 1,
        'memory_usage': 0.8,
        'created_by': 'system',
        'version': '1.0'
    },
    'theater_custom': {
        'name': 'Custom Theater Settings',
        'description': 'Customizable preset for specific requirements',
        'theater_size': 'medium',
        'quality_level': 'balanced',
        'target_resolution': '3840x2160',
        'denoise_strength': 0.7,
        'dialogue_boost': 4.0,
        'noise_reduction': 0.5,
        'reverb_reduction': 0.6,
        'preserve_naturalness': True,
        'tile_size': 768,
        'batch_size': 2,
        'memory_usage': 0.7,
        'created_by': 'user',
        'version': '1.0'
    }
}


def get_default_config() -> Dict[str, Any]:
    """
    Get a copy of the default configuration.
    
    Returns:
        Deep copy of default configuration dictionary
    """
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)


def get_default_preset(preset_name: str) -> Dict[str, Any]:
    """
    Get a copy of a default preset configuration.
    
    Args:
        preset_name: Name of preset to retrieve
        
    Returns:
        Deep copy of preset configuration dictionary
        
    Raises:
        KeyError: If preset name is not found
    """
    import copy
    
    if preset_name not in DEFAULT_PRESETS:
        available = list(DEFAULT_PRESETS.keys())
        raise KeyError(f"Preset '{preset_name}' not found. Available presets: {available}")
    
    return copy.deepcopy(DEFAULT_PRESETS[preset_name])


def get_available_presets() -> Dict[str, str]:
    """
    Get list of available default presets with descriptions.
    
    Returns:
        Dictionary mapping preset names to descriptions
    """
    return {
        name: preset['description'] 
        for name, preset in DEFAULT_PRESETS.items()
    }


def get_quality_levels() -> Dict[str, Dict[str, Any]]:
    """
    Get available quality levels with their parameters.
    
    Returns:
        Dictionary mapping quality levels to their parameters
    """
    return DEFAULT_CONFIG['video']['quality_presets'].copy()


def get_theater_sizes() -> Dict[str, Dict[str, Any]]:
    """
    Get available theater sizes with their audio parameters.
    
    Returns:
        Dictionary mapping theater sizes to their audio parameters
    """
    return DEFAULT_CONFIG['audio']['theater_presets'].copy()


def validate_preset_compatibility(preset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate preset compatibility with current system.
    
    Args:
        preset: Preset configuration to validate
        
    Returns:
        Dictionary containing validation results and recommendations
    """
    result = {
        'compatible': True,
        'warnings': [],
        'recommendations': []
    }
    
    try:
        # Check memory requirements
        memory_usage = preset.get('memory_usage', 0.7)
        tile_size = preset.get('tile_size', 768)
        batch_size = preset.get('batch_size', 2)
        
        # Estimate memory requirement (rough calculation)
        estimated_memory_mb = (tile_size * tile_size * 3 * batch_size * 4) // (1024 * 1024)
        
        # Get system memory
        try:
            from ..utils.system_utils import get_memory_info
            memory_info = get_memory_info()
            available_memory = memory_info.get('available_mb', 0)
            
            if estimated_memory_mb > available_memory * memory_usage:
                result['warnings'].append(
                    f"Preset may require more memory than available "
                    f"(estimated: {estimated_memory_mb}MB, available: {int(available_memory * memory_usage)}MB)"
                )
                result['recommendations'].append("Consider using a lower quality preset or smaller tile size")
        
        except ImportError:
            result['warnings'].append("Cannot verify memory compatibility")
        
        # Check GPU requirements
        quality_level = preset.get('quality_level', 'balanced')
        if quality_level == 'high':
            try:
                from ..utils.system_utils import check_gpu_availability
                gpu_info = check_gpu_availability()
                
                if not gpu_info.get('gpu_available', False):
                    result['warnings'].append("High quality preset recommended with GPU acceleration")
                    result['recommendations'].append("Consider using 'balanced' quality for CPU-only processing")
                elif gpu_info.get('gpu_memory_mb', 0) < 4000:  # Less than 4GB GPU
                    result['warnings'].append("High quality preset may be slow with limited GPU memory")
                    result['recommendations'].append("Consider reducing tile size or batch size")
            
            except ImportError:
                result['warnings'].append("Cannot verify GPU compatibility")
        
        # Check theater size compatibility
        theater_size = preset.get('theater_size', 'medium')
        dialogue_boost = preset.get('dialogue_boost', 4.0)
        
        if theater_size == 'large' and dialogue_boost > 4.0:
            result['recommendations'].append(
                "Large theater preset with high dialogue boost may sound unnatural"
            )
        
    except Exception as e:
        result['compatible'] = False
        result['warnings'].append(f"Error validating preset compatibility: {e}")
    
    return result