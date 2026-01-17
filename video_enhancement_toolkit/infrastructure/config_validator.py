"""
Configuration Validation Utilities

Provides validation functions for configuration parameters.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple


class ConfigValidator:
    """Utility class for configuration validation."""
    
    @staticmethod
    def validate_file_path(path: str, must_exist: bool = False) -> Tuple[bool, str]:
        """Validate file path.
        
        Args:
            path: File path to validate
            must_exist: Whether file must already exist
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not path or not isinstance(path, str):
            return False, "Path must be a non-empty string"
        
        try:
            path_obj = Path(path)
            
            if must_exist and not path_obj.exists():
                return False, f"File does not exist: {path}"
            
            # Check if parent directory exists or can be created
            parent = path_obj.parent
            if not parent.exists():
                try:
                    parent.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    return False, f"Cannot create parent directory: {e}"
            
            return True, ""
            
        except (OSError, ValueError) as e:
            return False, f"Invalid path: {e}"
    
    @staticmethod
    def validate_resolution(resolution: List[int]) -> Tuple[bool, str]:
        """Validate video resolution.
        
        Args:
            resolution: List of [width, height]
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(resolution, list) or len(resolution) != 2:
            return False, "Resolution must be a list of [width, height]"
        
        width, height = resolution
        
        if not isinstance(width, int) or not isinstance(height, int):
            return False, "Width and height must be integers"
        
        if width <= 0 or height <= 0:
            return False, "Width and height must be positive"
        
        # Common video resolutions validation
        common_resolutions = [
            (1280, 720),   # 720p
            (1920, 1080),  # 1080p
            (2560, 1440),  # 1440p
            (3840, 2160),  # 4K
            (7680, 4320),  # 8K
        ]
        
        # Allow custom resolutions but warn if not standard
        if tuple(resolution) not in common_resolutions:
            # Check if aspect ratio is reasonable (between 1:3 and 3:1)
            aspect_ratio = width / height
            if aspect_ratio < 1/3 or aspect_ratio > 3:
                return False, f"Unusual aspect ratio: {aspect_ratio:.2f}"
        
        return True, ""
    
    @staticmethod
    def validate_ai_model(model: str) -> Tuple[bool, str]:
        """Validate AI model name.
        
        Args:
            model: AI model name
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not model or not isinstance(model, str):
            return False, "Model name must be a non-empty string"
        
        supported_models = [
            "realesrgan-x4plus",
            "realesrgan-x4plus-anime",
            "real-cugan",
            "waifu2x-cunet",
            "waifu2x-photo",
            "esrgan-x4",
            "srmd-ncnn"
        ]
        
        if model not in supported_models:
            return False, f"Unsupported model: {model}. Supported models: {', '.join(supported_models)}"
        
        return True, ""
    
    @staticmethod
    def validate_theater_preset(preset: str) -> Tuple[bool, str]:
        """Validate theater preset name.
        
        Args:
            preset: Theater preset name
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not preset or not isinstance(preset, str):
            return False, "Preset name must be a non-empty string"
        
        valid_presets = ["small", "medium", "large"]
        
        if preset not in valid_presets:
            return False, f"Invalid theater preset: {preset}. Valid presets: {', '.join(valid_presets)}"
        
        return True, ""
    
    @staticmethod
    def validate_performance_config(config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate performance configuration.
        
        Args:
            config: Performance configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        cpu_threads = config.get("cpu_threads", 1)
        gpu_memory_limit = config.get("gpu_memory_limit", 1.0)
        
        # Validate CPU threads
        if not isinstance(cpu_threads, int) or cpu_threads <= 0:
            return False, "CPU threads must be a positive integer"
        
        # Check against system capabilities
        import os
        max_threads = os.cpu_count() or 1
        if cpu_threads > max_threads * 2:  # Allow hyperthreading
            return False, f"CPU threads ({cpu_threads}) exceeds system capability ({max_threads * 2})"
        
        # Validate GPU memory limit
        if not isinstance(gpu_memory_limit, (int, float)) or gpu_memory_limit <= 0:
            return False, "GPU memory limit must be a positive number"
        
        if gpu_memory_limit > 64:  # Reasonable upper limit
            return False, "GPU memory limit seems too high (>64GB)"
        
        return True, ""
    
    @staticmethod
    def validate_audio_config(config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate audio configuration.
        
        Args:
            config: Audio configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        noise_reduction = config.get("noise_reduction_strength", 0.3)
        dialogue_enhancement = config.get("dialogue_enhancement", 0.4)
        dynamic_range = config.get("dynamic_range_target", -23.0)
        
        # Validate noise reduction strength
        if not isinstance(noise_reduction, (int, float)) or not 0.0 <= noise_reduction <= 1.0:
            return False, "Noise reduction strength must be between 0.0 and 1.0"
        
        # Validate dialogue enhancement
        if not isinstance(dialogue_enhancement, (int, float)) or not 0.0 <= dialogue_enhancement <= 1.0:
            return False, "Dialogue enhancement must be between 0.0 and 1.0"
        
        # Validate dynamic range target
        if not isinstance(dynamic_range, (int, float)) or not -60.0 <= dynamic_range <= 0.0:
            return False, "Dynamic range target must be between -60.0 and 0.0 dB"
        
        # Check for over-processing
        if noise_reduction > 0.7 and dialogue_enhancement > 0.7:
            return False, "High noise reduction and dialogue enhancement may cause over-processing"
        
        return True, ""