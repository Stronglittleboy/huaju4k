"""
Validation utilities for huaju4k video enhancement.

This module provides validation functions for input files, configurations,
and system requirements.
"""

import os
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Supported video formats
SUPPORTED_VIDEO_FORMATS = {
    '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg'
}

# Supported audio formats  
SUPPORTED_AUDIO_FORMATS = {
    '.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a', '.wma'
}

# Minimum system requirements
MIN_SYSTEM_REQUIREMENTS = {
    'memory_mb': 4000,      # 4GB RAM minimum
    'disk_space_mb': 10000, # 10GB free space minimum
    'cpu_cores': 2          # 2 CPU cores minimum
}


def validate_video_file(file_path: str) -> Dict[str, Any]:
    """
    Validate video file for processing.
    
    Args:
        file_path: Path to video file
        
    Returns:
        Dictionary containing validation results
    """
    result = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'file_info': {}
    }
    
    try:
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            result['errors'].append(f"File does not exist: {file_path}")
            return result
        
        # Check if it's a file (not directory)
        if not path.is_file():
            result['errors'].append(f"Path is not a file: {file_path}")
            return result
        
        # Check file extension
        file_extension = path.suffix.lower()
        if file_extension not in SUPPORTED_VIDEO_FORMATS:
            result['errors'].append(f"Unsupported video format: {file_extension}")
            return result
        
        # Check file size
        file_size = path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        result['file_info'] = {
            'path': str(path.absolute()),
            'size_bytes': file_size,
            'size_mb': file_size_mb,
            'extension': file_extension,
            'name': path.stem
        }
        
        # Check if file is too small (likely corrupted)
        if file_size < 1024:  # Less than 1KB
            result['errors'].append("File is too small, possibly corrupted")
            return result
        
        # Check if file is extremely large
        if file_size_mb > 50000:  # More than 50GB
            result['warnings'].append("File is very large, processing may take a long time")
        
        # Check file permissions
        if not os.access(path, os.R_OK):
            result['errors'].append("File is not readable, check permissions")
            return result
        
        # Try to get MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and not mime_type.startswith('video/'):
            result['warnings'].append(f"MIME type suggests non-video file: {mime_type}")
        
        result['valid'] = True
        
    except Exception as e:
        result['errors'].append(f"Error validating file: {str(e)}")
        logger.error(f"Error validating video file {file_path}: {e}")
    
    return result


def validate_output_path(output_path: str, input_path: str = None) -> Dict[str, Any]:
    """
    Validate output path for video processing.
    
    Args:
        output_path: Desired output file path
        input_path: Optional input file path for comparison
        
    Returns:
        Dictionary containing validation results
    """
    result = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'path_info': {}
    }
    
    try:
        path = Path(output_path)
        
        # Check if parent directory exists or can be created
        parent_dir = path.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                result['errors'].append(f"Cannot create output directory: {e}")
                return result
        
        # Check if parent directory is writable
        if not os.access(parent_dir, os.W_OK):
            result['errors'].append("Output directory is not writable")
            return result
        
        # Check file extension
        file_extension = path.suffix.lower()
        if file_extension not in SUPPORTED_VIDEO_FORMATS:
            result['warnings'].append(f"Output format {file_extension} may not be optimal")
        
        # Check if output file already exists
        if path.exists():
            result['warnings'].append("Output file already exists and will be overwritten")
        
        # Check if input and output are the same
        if input_path:
            input_abs = Path(input_path).absolute()
            output_abs = path.absolute()
            if input_abs == output_abs:
                result['errors'].append("Input and output paths cannot be the same")
                return result
        
        # Check available disk space
        try:
            import shutil
            free_space = shutil.disk_usage(parent_dir).free
            free_space_mb = free_space / (1024 * 1024)
            
            result['path_info'] = {
                'path': str(path.absolute()),
                'parent_dir': str(parent_dir.absolute()),
                'extension': file_extension,
                'exists': path.exists(),
                'free_space_mb': free_space_mb
            }
            
            # Warn if low disk space
            if free_space_mb < 5000:  # Less than 5GB
                result['warnings'].append("Low disk space available for output")
            
        except Exception as e:
            result['warnings'].append(f"Could not check disk space: {e}")
        
        result['valid'] = True
        
    except Exception as e:
        result['errors'].append(f"Error validating output path: {str(e)}")
        logger.error(f"Error validating output path {output_path}: {e}")
    
    return result


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Dictionary containing validation results
    """
    result = {
        'valid': False,
        'errors': [],
        'warnings': []
    }
    
    try:
        # Check required sections
        required_sections = ['video', 'audio', 'performance']
        for section in required_sections:
            if section not in config:
                result['errors'].append(f"Missing required configuration section: {section}")
        
        if result['errors']:
            return result
        
        # Validate video configuration
        video_config = config.get('video', {})
        if 'ai_model' not in video_config:
            result['errors'].append("Missing ai_model in video configuration")
        
        if 'quality_presets' not in video_config:
            result['errors'].append("Missing quality_presets in video configuration")
        else:
            presets = video_config['quality_presets']
            required_presets = ['fast', 'balanced', 'high']
            for preset in required_presets:
                if preset not in presets:
                    result['warnings'].append(f"Missing quality preset: {preset}")
        
        # Validate audio configuration
        audio_config = config.get('audio', {})
        if 'theater_presets' not in audio_config:
            result['errors'].append("Missing theater_presets in audio configuration")
        else:
            presets = audio_config['theater_presets']
            required_presets = ['small', 'medium', 'large']
            for preset in required_presets:
                if preset not in presets:
                    result['warnings'].append(f"Missing theater preset: {preset}")
        
        # Validate performance configuration
        perf_config = config.get('performance', {})
        
        # Check memory usage setting
        max_memory = perf_config.get('max_memory_usage', 0.7)
        if not 0.1 <= max_memory <= 0.9:
            result['errors'].append("max_memory_usage must be between 0.1 and 0.9")
        
        # Check worker count
        num_workers = perf_config.get('num_workers', 2)
        if num_workers < 1 or num_workers > 16:
            result['warnings'].append("num_workers should be between 1 and 16")
        
        # Check GPU settings
        use_gpu = perf_config.get('use_gpu', True)
        if use_gpu:
            # Check if GPU is actually available
            try:
                from ..utils.system_utils import check_gpu_availability
                gpu_info = check_gpu_availability()
                if not gpu_info.get('gpu_available', False):
                    result['warnings'].append("GPU enabled but no GPU detected")
            except ImportError:
                result['warnings'].append("Cannot verify GPU availability")
        
        result['valid'] = len(result['errors']) == 0
        
    except Exception as e:
        result['errors'].append(f"Error validating configuration: {str(e)}")
        logger.error(f"Error validating configuration: {e}")
    
    return result


def is_supported_format(file_path: str, format_type: str = 'video') -> bool:
    """
    Check if file format is supported.
    
    Args:
        file_path: Path to file
        format_type: Type of format to check ('video' or 'audio')
        
    Returns:
        True if format is supported
    """
    try:
        extension = Path(file_path).suffix.lower()
        
        if format_type == 'video':
            return extension in SUPPORTED_VIDEO_FORMATS
        elif format_type == 'audio':
            return extension in SUPPORTED_AUDIO_FORMATS
        else:
            return False
            
    except Exception as e:
        logger.error(f"Error checking format support for {file_path}: {e}")
        return False


def validate_system_requirements() -> Dict[str, Any]:
    """
    Validate system meets minimum requirements.
    
    Returns:
        Dictionary containing validation results
    """
    result = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'system_info': {}
    }
    
    try:
        from ..utils.system_utils import get_system_info, get_memory_info, check_disk_space
        
        # Get system information
        system_info = get_system_info()
        memory_info = get_memory_info()
        
        result['system_info'] = {
            'cpu_cores': system_info.get('cpu_cores', 0),
            'total_memory_mb': system_info.get('total_memory_mb', 0),
            'available_memory_mb': memory_info.get('available_mb', 0),
            'gpu_available': system_info.get('gpu_available', False)
        }
        
        # Check CPU cores
        cpu_cores = system_info.get('cpu_cores', 0)
        if cpu_cores < MIN_SYSTEM_REQUIREMENTS['cpu_cores']:
            result['errors'].append(f"Insufficient CPU cores: {cpu_cores} (minimum: {MIN_SYSTEM_REQUIREMENTS['cpu_cores']})")
        
        # Check memory
        total_memory = system_info.get('total_memory_mb', 0)
        if total_memory < MIN_SYSTEM_REQUIREMENTS['memory_mb']:
            result['errors'].append(f"Insufficient memory: {total_memory}MB (minimum: {MIN_SYSTEM_REQUIREMENTS['memory_mb']}MB)")
        
        available_memory = memory_info.get('available_mb', 0)
        if available_memory < MIN_SYSTEM_REQUIREMENTS['memory_mb'] // 2:
            result['warnings'].append(f"Low available memory: {available_memory}MB")
        
        # Check disk space (current directory)
        try:
            disk_info = check_disk_space('.')
            free_space = disk_info.get('free_mb', 0)
            if free_space < MIN_SYSTEM_REQUIREMENTS['disk_space_mb']:
                result['errors'].append(f"Insufficient disk space: {free_space}MB (minimum: {MIN_SYSTEM_REQUIREMENTS['disk_space_mb']}MB)")
        except Exception as e:
            result['warnings'].append(f"Could not check disk space: {e}")
        
        # Check dependencies
        try:
            from ..utils.system_utils import check_dependencies
            deps = check_dependencies()
            
            critical_deps = ['ffmpeg', 'opencv', 'numpy']
            for dep in critical_deps:
                if not deps.get(dep, False):
                    result['errors'].append(f"Missing critical dependency: {dep}")
            
            optional_deps = ['torch', 'librosa']
            for dep in optional_deps:
                if not deps.get(dep, False):
                    result['warnings'].append(f"Missing optional dependency: {dep}")
                    
        except Exception as e:
            result['warnings'].append(f"Could not check dependencies: {e}")
        
        result['valid'] = len(result['errors']) == 0
        
    except Exception as e:
        result['errors'].append(f"Error validating system requirements: {str(e)}")
        logger.error(f"Error validating system requirements: {e}")
    
    return result


def validate_processing_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate processing parameters.
    
    Args:
        params: Processing parameters dictionary
        
    Returns:
        Dictionary containing validation results
    """
    result = {
        'valid': False,
        'errors': [],
        'warnings': []
    }
    
    try:
        # Check tile size
        tile_size = params.get('tile_size', 512)
        if not isinstance(tile_size, int) or tile_size < 64 or tile_size > 2048:
            result['errors'].append("tile_size must be integer between 64 and 2048")
        
        # Check batch size
        batch_size = params.get('batch_size', 1)
        if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 16:
            result['errors'].append("batch_size must be integer between 1 and 16")
        
        # Check denoise strength
        denoise = params.get('denoise_strength', 0.5)
        if not isinstance(denoise, (int, float)) or denoise < 0.0 or denoise > 1.0:
            result['errors'].append("denoise_strength must be number between 0.0 and 1.0")
        
        # Check quality preset
        quality = params.get('quality_preset', 'balanced')
        valid_qualities = ['fast', 'balanced', 'high']
        if quality not in valid_qualities:
            result['errors'].append(f"quality_preset must be one of {valid_qualities}")
        
        # Check theater preset
        theater = params.get('theater_preset', 'medium')
        valid_theaters = ['small', 'medium', 'large']
        if theater not in valid_theaters:
            result['errors'].append(f"theater_preset must be one of {valid_theaters}")
        
        result['valid'] = len(result['errors']) == 0
        
    except Exception as e:
        result['errors'].append(f"Error validating parameters: {str(e)}")
        logger.error(f"Error validating processing parameters: {e}")
    
    return result